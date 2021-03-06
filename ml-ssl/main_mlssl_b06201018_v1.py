# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex
from apex.parallel.LARC import LARC

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
from src.jigsawdataset_b06201018_v1 import JigsawDataset
import src.resnet50_b06201018_v1 as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--nmb_patch_per_size", type=int, default=3,
                    help="number of local patches per side of an image")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_gbl_prototypes", default=3000, type=int,
                    help="number of global prototypes")
parser.add_argument("--nmb_lcl_prototypes", default=5000, type=int,
                    help="number of local prototypes")
parser.add_argument("--gbl_queue_length", type=int, default=0,
                    help="length of the global queue (0 for no queue)")
parser.add_argument("--lcl_queue_length", type=int, default=0,
                    help="length of the local queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")


def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    train_dataset = JigsawDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        num_patch_per_side=args.nmb_patch_per_size
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_gbl_prototypes=args.nmb_gbl_prototypes,
        nmb_lcl_prototypes=args.nmb_lcl_prototypes,
    )
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on]
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        amp=apex.amp,
    )
    start_epoch = to_restore["epoch"]

    # build the global queue
    gbl_queue = None
    gbl_queue_path = os.path.join(args.dump_path, "gbl_queue" + str(args.rank) + ".pth")
    if os.path.isfile(gbl_queue_path):
        gbl_queue = torch.load(gbl_queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.gbl_queue_length -= args.gbl_queue_length % (args.batch_size * args.world_size)

    # build the local queue
    lcl_queue = None
    lcl_queue_path = os.path.join(args.dump_path, "lcl_queue" + str(args.rank) + ".pth")
    if os.path.isfile(lcl_queue_path):
        lcl_queue = torch.load(lcl_queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.lcl_queue_length -= args.lcl_queue_length % (args.batch_size * args.world_size)

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.gbl_queue_length > 0 and epoch >= args.epoch_queue_starts and gbl_queue is None:
            gbl_queue = torch.zeros(
                len(args.crops_for_assign),
                args.gbl_queue_length // args.world_size,
                args.feat_dim,
            ).cuda()
        if args.lcl_queue_length > 0 and epoch >= args.epoch_queue_starts and lcl_queue is None:
            lcl_queue = torch.zeros(
                len(args.crops_for_assign),
                args.lcl_queue_length // args.world_size,
                args.feat_dim,
            ).cuda()

        # train the network
        scores, gbl_queue, lcl_queue = train(train_loader, model, optimizer, epoch, lr_schedule, gbl_queue, lcl_queue)
        training_stats.update(scores)

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                save_dict["amp"] = apex.amp.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        if gbl_queue is not None:
            torch.save({"queue": gbl_queue}, gbl_queue_path)
        if lcl_queue is not None:
            torch.save({"queue": lcl_queue}, lcl_queue_path)


def train(train_loader, model, optimizer, epoch, lr_schedule, gbl_queue, lcl_queue):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    gbl_losses = AverageMeter()
    lcl_losses = AverageMeter()

    model.train()
    use_the_queue = False

    end = time.time()
    for it, inputs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = model.module.gbl_prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.gbl_prototypes.weight.copy_(w)
            w = model.module.lcl_prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.lcl_prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        (gbl_emb, gbl_out), (lcl_emb, lcl_out) = model(inputs)
        gbl_emb = gbl_emb.detach()
        lcl_emb = lcl_emb.detach()
        bs = inputs[0].size(0)

        # ============ global swav loss ... ============
        gbl_loss = 0
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = gbl_out[bs * crop_id: bs * (crop_id + 1)].detach()

                # time to use the queue
                if gbl_queue is not None:
                    if use_the_queue or not torch.all(gbl_queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            gbl_queue[i],
                            model.module.gbl_prototypes.weight.t()
                        ), out))
                    # fill the queue
                    gbl_queue[i, bs:] = gbl_queue[i, :-bs].clone()
                    gbl_queue[i, :bs] = gbl_emb[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = distributed_sinkhorn(out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                x = gbl_out[bs * v: bs * (v + 1)] / args.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            gbl_loss += subloss / (np.sum(args.nmb_crops) - 1)
        gbl_loss /= len(args.crops_for_assign)

        # ============ local swav loss ... ============
        n_patch = lcl_out.size(0) // gbl_out.size(0)
        lcl_bs = bs * n_patch
        lcl_loss = 0
        lcl_out = lcl_out.reshape(len(args.crops_for_assign), lcl_bs, -1)
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = lcl_out[crop_id].detach()

                # time to use the queue
                if lcl_queue is not None:
                    if use_the_queue or not torch.all(lcl_queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            lcl_queue[i],
                            model.module.lcl_prototypes.weight.t()
                        ), out))
                    # fill the queue
                    lcl_queue[i, lcl_bs:] = lcl_queue[i, :-lcl_bs].clone()
                    lcl_queue[i, :lcl_bs] = lcl_emb[crop_id * lcl_bs: (crop_id + 1) * lcl_bs]

                # get assignments
                q = distributed_sinkhorn(out)[-lcl_bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                x = lcl_out[v] / args.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            lcl_loss += subloss / (np.sum(args.nmb_crops) - 1)
        lcl_loss /= len(args.crops_for_assign)

        loss = gbl_loss + lcl_loss

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # cancel gradients for the prototypes
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        gbl_losses.update(gbl_loss.item(), inputs[0].size(0))
        lcl_losses.update(lcl_loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "GBL {gbl_loss.val:.4f} ({gbl_loss.avg:.4f})\t"
                "LCL {lcl_loss.val:.4f} ({lcl_loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    gbl_loss=gbl_losses,
                    lcl_loss=lcl_losses,
                    lr=optimizer.optim.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.avg), gbl_queue, lcl_queue


@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


if __name__ == "__main__":
    main()

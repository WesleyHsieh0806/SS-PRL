# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import ntpath
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
    concat_local_logits,
)
from src.localpatch_dataset import JigsawDataset
import src.resnet50 as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of ML-SSL")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--nmb_loc_views", type=int, default=[2], nargs="+",
                    help="number of views for each local patch (example: [2])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--loc_size_crops", type=int, default=[255], nargs="+",
                    help="views for local patches resolutions (example: [255])")
parser.add_argument("--loc_size_patches", type=int, default=[64], nargs="+",
                    help="local patch size (example: [64])")
parser.add_argument("--grid_perside", type=int, default=[3], nargs="+",
                    help="Number of grids per side for local images. (example: [3])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14 0.05 0.6])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1. 0.14 1.])")

#########################
## MLSSL specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--loc_view_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of local view id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_ptypes", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--nmb_local_ptypes", type=int, default=[5000], nargs="+",
                    help="number of local prototypes (example: [5000])")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--local_queue_length", type=int, default=[0], nargs="+",
                    help="length of the queue for local patches(0 for no queue) (example: [0])")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")
parser.add_argument("--lambda1", type=float, default=[0.5], nargs="+",
                    help="coefficient for the loc_loss")
parser.add_argument("--lambda2", type=float, default=[0.5], nargs="+",
                    help="coefficient for the l2g_loss")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float,
                    help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0,
                    help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10,
                    type=int, help="number of warmup epochs")
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
parser.add_argument("--arch", default="resnet50",
                    type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str,
                    default="pytorch", help="synchronize bn")
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
    # logger: such as log files and consoles training_stats: save logs into pickle files
    loss_to_track = ["glb"]
    for i in range(len(args.nmb_local_ptypes)):
        loss_to_track.append(f"loc{i}")
    for i in range(len(args.nmb_local_ptypes)):
        loss_to_track.append(f"l2g{i}")
    logger, training_stats = initialize_exp(
        args, "epoch", "loss", *loss_to_track)

    # build data
    train_dataset = JigsawDataset(
        args.data_path,
        args.size_crops,
        args.loc_size_crops,
        args.loc_size_patches,
        args.nmb_crops,
        args.nmb_loc_views,
        args.min_scale_crops,
        args.max_scale_crops,
        args.grid_perside
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
    logger.info("Building data done with {} images loaded.".format(
        len(train_dataset)))

    # build model
    # .__dict__ returns the dictionary of this package, all the functions/classes can be called by .__dict__[func name]
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_ptypes=args.nmb_ptypes,
        nmb_local_ptypes=args.nmb_local_ptypes,
        npatch=[0],
        grid_per_side=args.grid_perside,
    )
    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(
            args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(
            model, process_group=process_group)
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
    warmup_lr_schedule = np.linspace(
        args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 +
                                                                                           math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(
            model, optimizer, opt_level="O1")
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

    ''' ************
    * Queues
    *****************
    '''
    # build the queue
    queue = None
    queue_path = os.path.join(
        args.dump_path, "queue" + str(args.rank) + ".pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (
        args.batch_size * args.world_size)

    # build the local queue
    local_queues = []
    local_queue_paths = []
    for i in range(len(args.local_queue_length)):
        local_queue = None
        local_queue_path = os.path.join(
            args.dump_path, f"local_queue_{i}" + str(args.rank) + ".pth")
        if os.path.isfile(local_queue_path):
            local_queue = torch.load(local_queue_path)["local_queue"]
        # the queue needs to be divisible by the batch size
        args.local_queue_length[i] -= args.local_queue_length[i] % (
            args.batch_size * args.world_size)
        local_queues.append(local_queue)
        local_queue_paths.append(local_queue_path)

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        '''
        * Start Queues (Optional)
        '''
        # Global Queues
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()
        # Local Queues
        for i in range(len(args.local_queue_length)):
            if args.local_queue_length[i] > 0 and epoch >= args.epoch_queue_starts and local_queues[i] is None:
                local_queues[i] = torch.zeros(
                    len(args.loc_view_for_assign),
                    args.local_queue_length[i] // args.world_size,
                    args.feat_dim,
                ).cuda()

        # train the network
        scores, queue, local_queues = train(train_loader, model,
                                            optimizer, epoch, lr_schedule, queue, local_queues, args.lambda1, args.lambda2)
        # the scores include epoch and loss
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
                    os.path.join(args.dump_checkpoints,
                                 "ckp-" + str(epoch) + ".pth"),
                )
        if queue is not None:
            torch.save({"queue": queue}, queue_path)
        for i in range(len(args.local_queue_length)):
            if local_queues[i] is not None:
                torch.save(
                    {"local_queue": local_queues[i]}, local_queue_paths[i])


def train(train_loader, model, optimizer, epoch, lr_schedule, queue, local_queues, lambda1=[0.5], lambda2=[0.5]):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    glb_losses = AverageMeter()
    loc_losses = [AverageMeter() for _ in range(len(lambda1))]
    l2g_losses = [AverageMeter() for _ in range(len(lambda1))]

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
            # in DDP, use .module before calling functions
            model.module.ptypes_normalize()

        # ============ multi-res forward passes ... ============
        (glb_z, glb_logits), (loc_z_list, loc_logits_list) = model(inputs)
        glb_z = glb_z.detach()
        for i in range(len(loc_z_list)):
            loc_z_list[i] = loc_z_list[i].detach()
        bs = inputs[0].size(0)

        # ============ Global ML-SSL loss ... ============
        glb_loss = 0
        global_q = []
        for i, view_id in enumerate(args.crops_for_assign):
            # view_id is the index of view to be predicted
            with torch.no_grad():
                out = glb_logits[bs * view_id: bs * (view_id + 1)].detach()

                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[i],
                            model.module.ptypes.weight.t()
                        ), out))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = glb_z[view_id * bs: (view_id + 1) * bs]

                # get assignments
                q = distributed_sinkhorn(out)[-bs:]
                global_q.append(q)

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), view_id):
                x = glb_logits[bs * v: bs * (v + 1)] / args.temperature
                subloss -= torch.mean(torch.sum(q *
                                                F.log_softmax(x, dim=1), dim=1))
            glb_loss += subloss / (np.sum(args.nmb_crops) - 1)
        glb_loss /= len(args.crops_for_assign)

        loc_loss_list, l2g_loss_list = [], []
        # ============ Local ML-SSL loss ... ============
        for i in range(len(loc_logits_list)):
            loc_loss = 0
            n_patch = (loc_logits_list[i].shape[0]//bs)//2
            lbs = bs * n_patch

            # reshape the logits for later loss computation
            # original shape:(bs*npatch*2, nmb_locptypes)
            loc_logits_list[i] = loc_logits_list[i].reshape(
                [2, lbs, loc_logits_list[i].shape[-1]])

            for j, view_id in enumerate(args.loc_view_for_assign):
                with torch.no_grad():
                    out = loc_logits_list[i][view_id].detach()

                    # time to use the queue
                    if local_queues[i] is not None:
                        if use_the_queue or not torch.all(local_queues[i][j, -1, :] == 0):
                            use_the_queue = True
                            out = torch.cat((torch.mm(
                                local_queues[i][j],
                                getattr(model.module, "local_ptypes" +
                                        str(i)).weight.t()
                            ), out))
                        # fill the local_queue
                        local_queues[i][j,
                                        lbs:] = local_queues[i][j, :-lbs].clone()
                        local_queues[i][j, :lbs] = loc_z_list[i][view_id *
                                                                 lbs: (view_id + 1) * lbs]

                    # get assignments
                    loc_q = distributed_sinkhorn(out)[-lbs:]

                # cluster assignment prediction for local patches
                subloss = 0
                for v in np.delete(np.arange(np.sum(args.nmb_loc_views[i])), view_id):
                    # Notice that loc_logits[0][0~bs*n_patch-1] corresponds to loc_logits[1][0~bs*n_patch-1],
                    # so we can compare p,q directly
                    x = loc_logits_list[i][v] / args.temperature
                    subloss -= torch.mean(torch.sum(loc_q *
                                                    F.log_softmax(x, dim=1), dim=1))
                loc_loss += subloss / (np.sum(args.nmb_loc_views) - 1)
            loc_loss /= (len(args.loc_view_for_assign))
            loc_loss_list.append(loc_loss)

        # ============ Local2 Global ML-SSL loss ... ============
            l2g_loss = 0
            for v in np.arange(np.sum(args.nmb_loc_views[i])):
                # shape of loc_logits[v]: (batch*n_patch, 5000)
                # Average them up to (batch, 5000) and predict the global q
                mean_logits = concat_local_logits(
                    loc_logits_list[i][v], bs, n_patch)
                logits_l2g = model.module.forward_l2g(mean_logits, i)

                for g_vid in range(len(global_q)):
                    # Predict the clustering assignment of both gobal view
                    q_vid = global_q[g_vid]
                    p_l2g = logits_l2g / args.temperature
                    l2g_loss -= torch.mean(torch.sum(q_vid *
                                                     F.log_softmax(p_l2g, dim=1), dim=1))
            l2g_loss /= (np.sum(args.nmb_loc_views[i])*len(global_q))
            l2g_loss_list.append(l2g_loss)
        loc_loss_sum = sum([a*l for a, l in zip(lambda1, loc_loss_list)])
        l2g_loss_sum = sum([a*l for a, l in zip(lambda2, l2g_loss_list)])
        loss = glb_loss + loc_loss_sum + l2g_loss_sum
        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        if args.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # cancel gradients for the prototypes
        if iteration < args.freeze_prototypes_niters:
            model.module.clean_grad()
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        glb_losses.update(glb_loss.item(), inputs[0].size(0))
        for i in range(len(lambda1)):
            loc_losses[i].update(loc_loss_list[i].item(), inputs[0].size(0))
            l2g_losses[i].update(l2g_loss_list[i].item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0 and it % 50 == 0:
            log_info = (
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "GLB {glb_loss.val:.4f} ({glb_loss.avg:.4f})\t".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    glb_loss=glb_losses,
                )
            )
            for i in range(len(lambda1)):
                log_info += (
                    "LOC{0} {loc_loss.val:.4f} ({loc_loss.avg:.4f})\t".format(
                        i, loc_loss=loc_losses[i]))
            for i in range(len(lambda1)):
                log_info += (
                    "L2G{0} {l2g_loss.val:.4f} ({l2g_loss.avg:.4f})\t".format(
                        i, l2g_loss=l2g_losses[i]))
            log_info += (
                "Lr: {lr:.4f}".format(lr=optimizer.optim.param_groups[0]["lr"]))
            logger.info(log_info)

    scores = (epoch, losses.avg, glb_losses.avg)
    scores += tuple(l.avg for l in loc_losses)
    scores += tuple(l.avg for l in l2g_losses)

    return scores, queue, local_queue


@torch.no_grad()
def distributed_sinkhorn(out):
    # Q is K-by-B for consistency with notations from our paper
    Q = torch.exp(out / args.epsilon).t()
    B = Q.shape[1] * args.world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

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

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


if __name__ == "__main__":
    main()

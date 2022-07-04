# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
from random import shuffle
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim

from utils import (
    initialize_exp,
    fix_random_seeds,
    AverageMeter
)
from model import Backbone, RegLog
from dataset import COCODataset, VOCDataset
from metrics import compute_AP

logger = getLogger()


parser = argparse.ArgumentParser(description="Evaluate models: Multi-label classification")

#########################
#### main parameters ####
#########################
parser.add_argument("--task", default="linear", type=str, choices=["linear", "semisup"], 
                    help="which downstream task to evaluate")
parser.add_argument("--dataset", default="coco", type=str, choices=["coco", "voc"], 
                    help="on which dataset to evaluate the model")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=777, help="seed")
parser.add_argument("--data_path", type=str, default="/path/to/dataset",
                    help="path to dataset repository")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--labels_perc", type=str, default="10", choices=["1", "10"],
                    help="fine-tune on either 1% or 10% of labels. "
                         "used only in downstream semi-supervised training task")

#########################
#### model parameters ###
#########################
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model's checkpoint")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
parser.add_argument("--lr_last_layer", default=0.02, type=float, 
                    help="initial learning rate of the last fc layer. "
                         "used only in downstream semi-supervised training task")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--nesterov", action="store_true", help="nesterov momentum")
parser.add_argument("--scheduler_type", default="cosine", type=str, choices=["step", "cosine"])
# for multi-step learning rate decay
parser.add_argument("--decay_epochs", type=int, nargs="+", default=[60, 80],
                    help="Epochs at which to decay learning rate.")
parser.add_argument("--gamma", type=float, default=0.1, help="decay factor")
# for cosine learning rate schedule
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")


def main():
    args = parser.parse_args()
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(
        args, "epoch", "loss", "loss_val", "mAP_val"
    )

    # build data
    dataset = COCODataset if args.dataset == 'coco' else VOCDataset
    train_dataset = dataset(args.data_path, is_train=True)
    val_dataset = dataset(args.data_path, is_train=False)

    if args.task == 'semisup':
        indices = torch.load(f"indices_{args.labels_perc}perc.pth")
        train_sampler = torch.utils.data.SubsetRandomSampler(indices)
        shuffle = False
    elif args.task == 'linear':
        train_sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=train_sampler,
        shuffle=shuffle
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    logger.info("Building data done")

    # build model
    model = Backbone(args.pretrained)
    num_labels = 80 if args.dataset == 'coco' else 20
    linear_classifier = RegLog(num_labels)

    # model to gpu
    model = model.cuda()
    linear_classifier = linear_classifier.cuda()


    # set optimizer
    if args.task == 'semisup':
        optimizer = torch.optim.SGD(
            [{'params': model.parameters()},
            {'params': linear_classifier.parameters(), 'lr': args.lr_last_layer}],
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.wd,
        )
    elif args.task == 'linear':
        optimizer = torch.optim.SGD(
            linear_classifier.parameters(),
            lr=args.lr,
            nesterov=args.nesterov,
            momentum=0.9,
            weight_decay=args.wd,
        )

    # set scheduler
    if args.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, args.decay_epochs, gamma=args.gamma
        )
    elif args.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.final_lr
        )

    # Optionally resume from a checkpoint
    start_epoch = 0
    global best_mAP
    best_mAP = 0.

    ckpt_path = os.path.join(args.dump_path, "checkpoint.pth.tar")
    if os.path.isfile(ckpt_path):
        logger.info("Found checkpoint at {}".format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        linear_classifier.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint["epoch"]
        best_mAP = checkpoint["best_mAP"]
    
    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        scores = train(model, linear_classifier, optimizer, train_loader, epoch, args.task)
        scores_val = validate_network(val_loader, model, linear_classifier)
        training_stats.update(scores + scores_val)

        scheduler.step()

        # save checkpoint
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": linear_classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_mAP": best_mAP,
        }
        torch.save(save_dict, os.path.join(args.dump_path, "checkpoint.pth.tar"))

    logger.info("Training of the supervised linear classifier on frozen features completed.\n"
                "Test mAP: {mAP:.2f}".format(mAP=best_mAP))


def train(model, reglog, optimizer, loader, epoch, task):
    """
    Train the models on the dataset.
    """
    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    losses = AverageMeter()
    end = time.perf_counter()

    if task == 'linear':
        model.eval()
    elif task == 'semisup':
        model.train()
    reglog.train()
    criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        # move to gpu
        inp = inp.cuda()
        target = target.cuda()

        # forward
        with torch.no_grad():
            output = model(inp)
        output = reglog(output)

        # compute bce loss
        mask = (target == 255)
        loss = torch.sum(criterion(output, target).masked_fill_(mask, 0)) / target.size(0)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(reglog.parameters(), 10)

        # step
        optimizer.step()

        # update stats
        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if iter_epoch % 100 == 0:
            logger.info(
                "Epoch[{0}] - Iter: [{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "LR {lr}".format(
                    epoch,
                    iter_epoch,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

    return epoch, losses.avg


def validate_network(val_loader, model, linear_classifier):
    batch_time = AverageMeter()
    losses = AverageMeter()
    global best_mAP

    # switch to evaluate mode
    model.eval()
    linear_classifier.eval()

    criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()
    meta_predictions = None
    predictions = []
    labels = []

    with torch.no_grad():
        end = time.perf_counter()
        for i, (inp, target) in enumerate(val_loader):

            # move to gpu
            inp = inp.cuda()
            target = target.cuda()

            # compute output
            output = linear_classifier(model(inp))
            mask = (target == 255)
            loss = torch.sum(criterion(output, target).masked_fill_(mask, 0)) / target.size(0)

            losses.update(loss.item(), inp.size(0))
            predictions.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

    # compute mAP
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    APs, mAP = compute_AP(predictions, labels)

    if mAP > best_mAP:
        best_mAP = mAP

    logger.info(
        "Test:\t"
        "Time {batch_time.avg:.3f}\t"
        "Loss {loss.avg:.4f}\t"
        "mAP {mAP:.3f}\t"
        "Best mAP so far {best_mAP:.3f}".format(
            batch_time=batch_time, loss=losses, mAP=mAP, best_mAP=best_mAP))

    return losses.avg, mAP


if __name__ == "__main__":
    main()
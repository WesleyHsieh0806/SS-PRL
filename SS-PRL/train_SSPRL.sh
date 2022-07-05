#!/bin/bash

# This script gives a taste of how arguments should look like
# In short, several arguments related to local prototypes are changed to lists.

DATASET_PATH="/work/b06901053/SSPRL/COCO/tmp_root"
EXPERIMENT_PATH="./logs/"
mkdir -p $EXPERIMENT_PATH

EPOCH=200
BATCH_PERGPU=64
LAMBDA1="0.25 0.25"
LAMBDA2="0.25 0.25"

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502  main_SSPRL.py \
--data_path $DATASET_PATH \
--workers 4 \
--nmb_crops 2 6 \
--nmb_loc_views 2 2 \
--size_crops 224 96 \
--loc_size_crops 314 383 \
--loc_size_patches 112 96 \
--min_scale_crops 0.14 0.05 0.6 0.6 \
--max_scale_crops 1. 0.14 1. 1. \
--grid_perside 2 3 \
--batch_size $BATCH_PERGPU \
--crops_for_assign 0 1 \
--loc_view_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_ptypes 300 \
--nmb_local_ptypes 150 150 \
--queue_length 1920 \
--local_queue_length 5000 5000 \
--epoch_queue_starts 80 \
--lambda1 $LAMBDA1 \
--lambda2 $LAMBDA2 \
--epochs $EPOCH \
--base_lr 0.6 \
--final_lr 0.0006 \
--freeze_prototypes_niters 648 \
--warmup_epochs 0 \
--arch resnet50 \
--dump_path $EXPERIMENT_PATH \


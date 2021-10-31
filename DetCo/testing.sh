#!/bin/bash

DATASET_PATH="~/b06201018/Mini-ImageNet"
EXPERIMENT_PATH="./experiments/DetCo_test"
mkdir -p $EXPERIMENT_PATH
EPOCH=200
BATCH_PERGPU=8

python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 main_MLSSL.py \
--data_path $DATASET_PATH \
--nmb_crops 2 \
--nmb_loc_views 2 \
--size_crops 224 \
--loc_size_crops 255 \
--min_scale_crops 0.14 0.6 \
--max_scale_crops 1. 1. \
--workers 4 \
--batch_size $BATCH_PERGPU \
--crops_for_assign 0 1 \
--loc_view_for_assign 0 1 \
--epsilon 0.05 \
--feat_dim 128 \
--epochs $EPOCH \
--base_lr 0.6 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--arch resnet50 \
--use_fp16 true \
--grid_perside 3 \
--dump_path $EXPERIMENT_PATH \
--detco-t 0.07 \
--detco-dim 128 \
--detco-k 65536 \
--detco-m 0.999 \
--detco-weightslist 0.1 0.4 0.7 1.0
###



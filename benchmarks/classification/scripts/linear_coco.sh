#!/bin/bash
DATA=$1         # path to the coco dataset
PRETRAINED=$2   # path to the pre-trained weight

python main.py \
    --task linear \
    --dump_path logs/linear_coco \
    --dataset coco \
    --data_path ${DATA} \
    --pretrained ${PRETRAINED} \
    --batch_size 128 \
    --epochs 100 \
    --lr 0.1
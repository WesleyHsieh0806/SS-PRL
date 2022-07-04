#!/bin/bash
DATA=$1         # path to the voc dataset
PRETRAINED=$2   # path to the pre-trained weight

python main.py \
    --task linear \
    --dump_path logs/linear_voc \
    --dataset voc \
    --data_path ${DATA} \
    --pretrained ${PRETRAINED} \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.03
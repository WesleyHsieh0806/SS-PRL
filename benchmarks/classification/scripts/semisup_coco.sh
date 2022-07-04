#!/bin/bash
DATA=$1         # path to the coco dataset
PRETRAINED=$2   # path to the pre-trained weight
PERC=$3         # fine-tune with 1 or 10 percent of labels

python main.py \
    --task semisup \
    --dump_path logs/semisup_coco_${PERC}perc \
    --dataset coco \
    --data_path ${DATA} \
    --pretrained ${PRETRAINED} \
    --batch_size 32 \
    --epochs 20 \
    --scheduler_type step \
    --decay_epochs 12 16 \
    --gamma 0.2 \
    --lr 0.001 \
    --lr_last_layer 0.1 \
    --labels_perc ${PERC}
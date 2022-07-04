#!/bin/bash
WEIGHTS=$1 # path to the pre-trained weight (in detectron2's format)
OUT=$2     # path to the output directory

python ./train_net.py \
	--config-file ./configs/R_50_FPN_1x.yaml \
	--num-gpus 8 \
	MODEL.WEIGHTS ${WEIGHTS} \
	OUTPUT_DIR ${OUT}
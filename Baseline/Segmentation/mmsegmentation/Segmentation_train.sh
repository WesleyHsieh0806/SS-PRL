#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee

source activate b06901053
cd $PBS_O_WORKDIR
module load cuda/cuda-9.2/x86_64

# Modify this line
CONFIG_FILE=configs/densecl/fcn_r50-d8_512x512_20k_voc12aug.py
GPUS=4
OUTPUT_DIR=logs/swav/swav_800
./tools/dist_train.sh ${CONFIG_FILE} ${GPUS} --work-dir ${OUTPUT_DIR}
###


conda deactivate

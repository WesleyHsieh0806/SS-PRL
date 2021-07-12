#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee

############### 1. modify the name of vir-env
source activate b06901053_mmseg
cd $PBS_O_WORKDIR
module load cuda/cuda-9.2/x86_64

######################
WORK_DIR=Models
CHECKPOINT=${WORK_DIR}/DenseCL.pth
WEIGHT_FILE=${WORK_DIR}/Extracted_DenseCL.pth

python tools/extract_backbone_weights.py ${CHECKPOINT} ${WEIGHT_FILE}
###


conda deactivate

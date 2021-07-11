#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee

<<<<<<< HEAD
#source activate b06901053
#cd $PBS_O_WORKDIR
=======
############### 1. modify the name of vir-env
source activate b06901053_mmseg
cd $PBS_O_WORKDIR
>>>>>>> 633d51f181d37845ad1dee00475d21e9ba3c0703
module load cuda/cuda-9.2/x86_64

######################
CONFIG_FILE=configs/densecl/fcn_r50-d8_512x512_20k_voc12aug.py
GPUS=4
# 2. Change the output dir 3. Change the path of pretrained model in fcn_r50-d8.py
OUTPUT_DIR=logs/DenseCL
./tools/dist_train.sh ${CONFIG_FILE} ${GPUS} --work-dir ${OUTPUT_DIR}
###


#conda deactivate

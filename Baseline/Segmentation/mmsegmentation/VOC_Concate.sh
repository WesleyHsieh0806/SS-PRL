#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee

############### 1. modify the name of vir-env
source activate b06901053_mmseg
cd $PBS_O_WORKDIR
module load cuda/cuda-9.2/x86_64

######################
python tools/convert_datasets/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug --nproc 8
###


conda deactivate

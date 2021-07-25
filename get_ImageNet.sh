#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee
#PBS -l walltime=12:00:00

############### 1. modify the name of vir-env
source activate b06901053_mmseg
cd $PBS_O_WORKDIR
module load cuda/cuda-9.2/x86_64

######################
# VOC12
DATA="$1"
if [ "$DATA" == "" ]; then
    echo "Usage: bash ./get_ImageNet.sh YOUR_DATA_ROOT"
    exit
fi
# -P DIR --> save files in DIR
mkdir $DATA
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar -P $DATA
tar -xf $DATA/ILSVRC2012_img_train_t3.tar -C $DATA


###


conda deactivate

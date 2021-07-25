#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee
#PBS -l walltime=24:00:00
cd $PBS_O_WORKDIR
######################
# VOC12
DATA="./data"
if [ "$DATA" == "" ]; then
    echo "Usage: bash ./get_ImageNet.sh YOUR_DATA_ROOT"
    exit
fi
# -P DIR --> save files in DIR
mkdir $DATA
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -P $DATA
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -P $DATA
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz -P $DATA

mkdir $DATA/train
tar -xf $DATA/ILSVRC2012_img_train.tar -C $DATA/train
rm $DATA/ILSVRC2012_img_train.tar

mkdir $DATA/val
tar -xf $DATA/ILSVRC2012_img_val.tar -C $DATA/val
rm $DATA/ILSVRC2012_img_val.tar
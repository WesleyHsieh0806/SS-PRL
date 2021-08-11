#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee
#PBS -l walltime=24:00:00
########## Mini-ImageNet Dataset Generation ##########
# Please modify the environment
source activate b06201018_swav
cd $PBS_O_WORKDIR

# Specify the path to ImageNt/train
IMAGENET="ImageNet/train"

# Feel free to modify the data directory
DATA="./MiniImageNet"

# Generate Mini-ImageNet dataset
mkdir csv_files
wget https://raw.githubusercontent.com/yaoyao-liu/mini-imagenet-tools/main/csv_files/train.csv -P csv_files
wget https://raw.githubusercontent.com/yaoyao-liu/mini-imagenet-tools/main/csv_files/val.csv -P csv_files
wget https://raw.githubusercontent.com/yaoyao-liu/mini-imagenet-tools/main/csv_files/test.csv -P csv_files
python get_MiniImageNet.py --imagenet_dir $IMAGENET --output_dir $DATA

# Move images in val and test to train so that we can use the whole 60,000 images for training
mv $DATA/val/* $DATA/train/
mv $DATA/test/* $DATA/train/

rm -r csv_files

conda deactivate

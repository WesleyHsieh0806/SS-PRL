#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee
#PBS -l walltime=24:00:00
########## COCO Dataset Generation ##########
cd $PBS_O_WORKDIR

# Feel free to modify the data directory
DATA="./COCO2014"
mkdir $DATA
cd $DATA

# Download COCO from the official website
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

# Unzip the files
unzip train2014.zip
unzip val2014.zip
unzip annotations_trainval2014.zip

rm *.zip

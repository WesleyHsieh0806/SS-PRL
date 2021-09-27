#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee
#PBS -l walltime=24:00:00
########## VOC Dataset Generation ##########
cd $PBS_O_WORKDIR

# Feel free to modify the data directory
DATA="./VOC2007"

# Download VOC07 from the official website
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

# Untar the file
tar -xvf VOCtrainval_06-Nov-2007.tar

# Move the dataset to specified directory 
mv VOCdevkit/VOC2007 $DATA

rm -r VOCdevkit
rm VOCtrainval_06-Nov-2007.tar

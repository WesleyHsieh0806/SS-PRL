#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee
#PBS -l walltime=24:00:00
########## Mini-ImageNet Dataset Generation ##########
# Please modify the environment
source activate b06201018_swav
cd $PBS_O_WORKDIR

# Feel free to modify the data directory
DATA="./MiniImageNet"
mkdir $DATA

# Clone the repository
git clone https://github.com/yaoyao-liu/mini-imagenet-tools.git
cd mini-imagenet-tools

# Download its provided tar file
at-get a306397ccf9c2ead27155983c254227c0fd938e2

# Generate Mini-ImageNet dataset
python mini_imagenet_generator.py --tar_dir ILSVRC2012_img_train.tar --image_resize 0

# Move processed images to specified folder
cd ..
mv mini-imagenet-tools/processed_images $DATA
rm -rf mini-imagenet-tools/

conda deactivate

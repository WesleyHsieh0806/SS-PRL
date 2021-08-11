#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee
#PBS -l walltime=24:00:00
########## Mini-COCO Dataset Generation ##########
# Please modify the environment
source activate b06201018_swav
cd $PBS_O_WORKDIR

# Feel free to modify the data directory
DATA="./MiniCOCO"
mkdir $DATA

# Clone the repository
git clone https://github.com/giddyyupp/coco-minitrain.git
cd coco-minitrain

# Download the required json file
wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=\
$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1lezhgY4M_Ag13w0dEzQ7x_zQ_w0ohjin' \
-O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lezhgY4M_Ag13w0dEzQ7x_zQ_w0ohjin" \
-O instances_minitrain2017.json && rm -rf /tmp/cookies.txt

# Generate Mini-COCO dataset
mkdir data
python code/coco_download.py --output_dir data

# Place dataset under root folder
cd ..
mv coco-minitrain/data $DATA/data
rm -rf coco-minitrain/

conda deactivate

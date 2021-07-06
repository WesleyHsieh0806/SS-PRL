#!/bin/bash

DATA="$1"
if [ "$DATA" == "" ]; then
    echo "Usage: bash prepare_data/prepare_voc07_cls.sh YOUR_DATA_ROOT"
    exit
fi

VOC="$DATA/VOCdevkit/VOC2007/"

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -P $DATA
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar -P $DATA
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar -P $DATA
tar -xf $DATA/VOCtrainval_11-May-2012.tar -C $DATA
tar -xf $DATA/VOCtrainval_06-Nov-2007.tar -C $DATA
tar -xf $DATA/VOCtest_06-Nov-2007.tar -C $DATA

mkdir $VOC/Lists

awk 'NF{print $0 ".jpg"}' $VOC/ImageSets/Main/trainval.txt $VOC/ImageSets/Main/test.txt > $VOC/Lists/trainvaltest.txt

mkdir data/
ln -s $DATA/VOCdevkit data/

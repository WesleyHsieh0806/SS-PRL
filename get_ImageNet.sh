
######################
# VOC12
DATA="$1"
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

mkdir $DATA/val
tar -xf $DATA/ILSVRC2012_img_val.tar -C $DATA/val

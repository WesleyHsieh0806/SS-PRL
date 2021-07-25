
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

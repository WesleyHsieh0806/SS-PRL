# VOC12
DATA="$1"
if [ "$DATA" == "" ]; then
    echo "Usage: bash ./get_VOC.sh YOUR_DATA_ROOT"
    exit
fi
# -P DIR --> save files in DIR
#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -P $DATA
#wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz -P $DATA --no-check-certificate
#tar -xf $DATA/VOCtrainval_11-May-2012.tar -C $DATA
tar -zxvf $DATA/benchmark.tgz -C $DATA

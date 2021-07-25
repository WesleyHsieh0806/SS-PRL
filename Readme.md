# Usage
## Download Imagenet
1. Go to [Image-net.org](https://image-net.org/index.php) to obtain the following files
    * **ILSVRC2012_img_train.tar**
    * **ILSVRC2012_img_val.tar**
    * **ILSVRC2012_devkit_t12.tar.gz**
    * ILSVRC2012_img_train_t3.tar(Optional)
    * ILSVRC2012_devkit_t3.tar.gz(Optional)
2. Preprocess ImageNet validation set:

    ``` bash
    # Modify the DATA_ROOT in shell script
    bash imagenet_preprocess.sh ${DATA_ROOT}
    ```
## :notebook_with_decorative_cover: Prepare Dataset
-----
### :racehorse:COCO
1. Download COCO

    you can also modify the ***$DATA_ROOT*** [here](./get_COCO.sh)
    ```bash
    bash get_COCO.sh
    ```
2. Since COCO dataset does not store images in corresponding class directory, we have to add a fake root directory under train_2014
    ```bash
    cd $DATA_ROOT
    mkdir -p tmp_root
    mv train2014 tmp_root/train2014
    ```
    ```none
     DATA_ROOT(For COCO)
     ├── tmp_root
     │   ├── train2014
     │   │   ├── COCO_train2014_XXXXXXXXXX80.jpg.png
     │   │   ├── COCO_train2014_XXXXXXXXXX81.jpg.png   
     ├── val2014
     │   ├── COCO_val2014_XXXXXXXXXX01
     │   ├── COCO_val2014_XXXXXXXXXX02.png
     │   ├── COCO_val2014_XXXXXXXXXX03.png   
    ```
3. If the pre-trained framework utilizes the annotation .txt for training data (Like InsLoc), create it with [this file](./)
---
### :dog:ImageNet
Download Imagenet
1.  Obtain the following files at [Image-net.org](https://image-net.org/index.php)
    * **ILSVRC2012_img_train.tar**
    * **ILSVRC2012_img_val.tar**
    * **ILSVRC2012_devkit_t12.tar.gz**
2. Preprocess ImageNet validation set:

    ``` bash
    # Modify the DATA_ROOT in shell script
    bash imagenet_preprocess.sh ${DATA_ROOT}
    ```
3. The directory structure should look like this:
   ```none
   DATA_ROOT(For ImageNet)
   ├── train
   │   ├── nXXXXX
   │   │   ├── nXXXXX_01.png
   │   │   ├── nXXXXX_02.png   
   ├── val
   │   ├── nXXXXX
   │   │   ├── nXXXXX_01.png
   │   │   ├── nXXXXX_02.png   
   ```

# Prepare Dataset
## Mini-ImageNet
   Use and modify [get_MiniImageNet.sh](./get_MiniImageNet.sh) according to the descriptions

## Mini-COCO
Use and modify [get_MiniCOCO.sh](./get_MiniCOCO.sh) according to the descriptions
* Requirements: 
   * wget
   * pycocotools

## ImageNet
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
   DATA_ROOT
   ├── train
   │   ├── nXXXXX
   │   │   ├── nXXXXX_01.png
   │   │   ├── nXXXXX_02.png   
   ├── val
   │   ├── nXXXXX
   │   │   ├── nXXXXX_01.png
   │   │   ├── nXXXXX_02.png   
   ```

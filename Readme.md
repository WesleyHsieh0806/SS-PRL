## Prepare Dataset
### Mini-ImageNet
   Use and modify [get_MiniImageNet.sh](./get_MiniImageNet.sh) according to the descriptions

### Mini-COCO
Use and modify [get_MiniCOCO.sh](./get_MiniCOCO.sh) according to the descriptions
* Requirements: 
   * wget
   * pycocotools

### ImageNet
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

## Usage - Training
Todo

## Downstream tasks
1. Download the pretrained models

   We provide the checkpoint files of SS-PRL and other SoTA used in our experiments,
   including
   * [SwAV](https://github.com/facebookresearch/swav)
   * [MoCo](https://github.com/facebookresearch/moco)
   * [DenseCL](https://github.com/WXinlong/DenseCL)
   * [BYOL](https://github.com/deepmind/deepmind-research/tree/master/byol)
   * [InsLoc](https://github.com/limbo0000/InstanceLoc)
   * [MaskCo](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Self-Supervised_Visual_Representations_Learning_by_Contrastive_Mask_Prediction_ICCV_2021_paper.html)

   ``` bash
   # Download the checkpoints with this command
   bash get_premodels.sh
   ```
2. Transferring to Multi-Label Visual Analysis tasks:

   Please Refer to Readme files for [Classification](./benchmarks/classification), [Object-Detection](./benchmarks/detection), and [Semantic Segmentation]((./benchmarks/Segmentation)) tasks.

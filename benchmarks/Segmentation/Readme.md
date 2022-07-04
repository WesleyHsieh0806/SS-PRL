# Usage
## Get the pretrained models :fire:
1. Download the pretrained models 
    ```bash 
    ./get_premodel.sh
    ```
2. Change pretrained models into the same format as DenseCL 
(We use the same setting as DenseCL in Semantic Segmentation)

    The framework listed below are covered in this python script:
    * [SwAV](https://github.com/facebookresearch/swav)
    * [MoCo](https://github.com/facebookresearch/moco)
    * [DenseCL](https://github.com/WXinlong/DenseCL)
    * [BYOL](https://github.com/deepmind/deepmind-research/tree/master/byol)
    * [InsLoc](https://github.com/limbo0000/InstanceLoc)
    * [MaskCo](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Self-Supervised_Visual_Representations_Learning_by_Contrastive_Mask_Prediction_ICCV_2021_paper.html)
    * [DetCo](https://scholar.google.com/scholar_url?url=http://openaccess.thecvf.com/content/ICCV2021/html/Xie_DetCo_Unsupervised_Contrastive_Learning_for_Object_Detection_ICCV_2021_paper.html&hl=zh-TW&sa=T&oi=gsb&ct=res&cd=0&d=8124073977598762954&ei=pbbCYrbwIMKM6rQP4NW28AU&scisig=AAGBfm0HSaylYjW3Py2zZuwpBf9JdfVLNQ) (modify_DetCo_models.py)
    
    ```bash
    # In this python script, we remove weights of projection and prototype 
    # Also, all the unecessary module names are removed
    python modify_models.py --pretrained ${ORG_Model} --model ${Dense_Model} --newmodel ${MODIFY_Model}
    ```
    ***${ORG_Model}*** : path of the original pretrained model

    ***${Dense_Model}*** : path of the DenseCL pretrained model (this file is used for format checking)

    ***${MODIFY_Model}*** : path of the Modified models
-------
## Semantic Segmentation Finetuning

### VOC12+aug
1. First, get the dataset by
    ```bash 
    bash get_VOC.sh $DATAROOT
    ```
2. Organize the dataset structure as the same as [Reame.md](https://github.com/WXinlong/mmsegmentation/blob/master/docs/dataset_prepare.md)

    :warning: Remember to use the following command to concatenate **VOC_aug** and **VOC_2012**

    ```bash    
        cd mmsegmentation
        # --nproc means 8 process for conversion, which could be omitted as well.
        python tools/convert_datasets/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug --nproc 8
    ```
3. create links between mmsegmentation/data and your **$DATAROOT**
    ```bash    
        cd mmsegmentation
        ln -s $DATAROOT/VOCdevkit ./data
    ```
### Start FineTuning
You can also refer to [DenseCL](https://github.com/WXinlong/DenseCL/blob/main/benchmarks/detection/README.md)
1. **cd mmsegmentation**
2. Then, modify the pretrained model path in [configs](mmsegmentation/configs/densecl/fcn_r50-d8.py#L5)
3. Modify the settings in [shell scripts](mmsegmentation/Segmentation_train.sh)
4. Start Training
    ``` bash
    bash Segmentation_train.sh
    ```

# Usage
## Get the pretrained models :fire:
1. Download the pretrained models 
    ```bash 
    ./get_premodel.sh
    ```
2. Change swav pretrained models into the same format as DenseCL 
(We use the same setting as DenseCL in Semantic Segmentation)
    ```bash
    # In this python script, we remove weights of projection and prototype 
    # Also, all the prefix 'module' are removed
    python modify_swav_models.py --pretrained ${ORG_Model} --model ${Dense_Model} --newmodel ${MODIFY_Model}
    ```
    ***${ORG_Model}*** : path of the original pretrained model

    ***${Dense_Model}*** : path of the DenseCL pretrained model

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
        # --nproc means 8 process for conversion, which could be omitted as well.
        python tools/convert_datasets/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug --nproc 8
    ```
3. create links between mmsegmentation/data and your **$DATAROOT**
### Start FineTuning
You can also refer to [DenseCL](https://github.com/WXinlong/DenseCL/blob/main/benchmarks/detection/README.md)
1. **cd mmsegmentation**
2. Then, modify the pretrained model path in [configs](mmsegmentation/configs/densecl/fcn_r50-d8.py#L5)
3. Modify the settings in [shell scripts](mmsegmentation/Segmentation_train.sh)
4. Start Training
    ``` bash
    qsub Segmentation_train.sh
    ```

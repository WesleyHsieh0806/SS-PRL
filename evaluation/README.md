# Multi-label Evaluation

This code provides a PyTorch implementation for multi-label evaluation on [COCO](https://cocodataset.org/#home) and [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).

## Dataset preparation
Please download [COCO-2014](https://cocodataset.org/#download) and [VOC-2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) from the official websites.  
For those who prefer to download the datasets from command line, please refer to [get_COCO.sh](../get_COCO.sh) and [get_VOC.sh](../get_VOC.sh).

## Supported pretrained models 
This code only supports certain SSL-pretrained models. Belows are the supported models and where to get their pretrained weights:  
|[SwAV](https://github.com/facebookresearch/swav#model-zoo)|[BYOL](https://github.com/kakaobrain/scrl#results)|[MoCo-v2](https://github.com/WXinlong/DenseCL#models)|[SCRL](https://github.com/kakaobrain/scrl#results)|[DenseCL](https://github.com/WXinlong/DenseCL#models)|[DetCo](https://github.com/xieenze/DetCo#download-models)|
|----------------------------------------------------------|--------------------------------------------------|-----------------------------------------------------|--------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------|  

Note that currently the only accepted model backbone is ResNet-50.

## Evaluating models
To train a supervised linear classifier on frozen features/weights, run:
```python
python linear_eval.py \
--dataset voc \
--data_path /path/to/voc/dataset \
--model_name byol \
--pretrained /path/to/byol/pretrained/weights \
--lr 0.3 \
--epochs 100 \
--dump_path /path/to/logging/directory
```
For more training options, please refer to the code [here](linear_eval.py#L32).  
Note that  
- `--dataset` should be either `voc` or `coco`, and `--data_path` should specify the path to the dataset.
- the options of `--model_name` are `[byol, swav, densecl, scrl, moco, detco]`, and `--pretrained` should specify the path to the pretrained weights mentioned in [this section](#supported-pretrained-models).  

Belows are the examples to evaluate ML-SSL model on VOC and COCO dataset:  
```python
python linear_eval.py \
--dataset voc \
--data_path VOC2007 \
--model_name swav \
--pretrained checkpoint.pth.tar \
--lr 0.03 \
--epochs 100 \
--dump_path logs/voc/
```
```python
python linear_eval.py \
--dataset coco \
--data_path COCO \
--model_name swav \
--pretrained checkpoint.pth.tar \
--lr 0.1 \
--epochs 100 \
--dump_path logs/coco/
```

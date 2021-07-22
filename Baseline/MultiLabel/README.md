# Multi-label Evaluation

This code provides a PyTorch implementation for multi-label evaluation on [COCO](https://cocodataset.org/#home) and [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).

## Dataset preparation
Please download [COCO-2014](https://cocodataset.org/#download) and [VOC-2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) from the official websites.

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
--epochs 100
```
For more training options, please refer to the code [here](https://github.com/WesleyHsieh0806/ML-SSL/blob/559c1e938414f91240c9e51a2e053a25ae66691b/Baseline/MultiLabel/linear_eval.py#L32).  
Note that  
- `--dataset` should be either `voc` or `coco`, and `--data_path` should specify the path to the dataset.
- the options of `--model_name` are `[byol, swav, densecl, scrl, moco, detco]`, and `--pretrained` should specify the path to the pretrained weights mentioned in [this section](#supported-pretrained-models).

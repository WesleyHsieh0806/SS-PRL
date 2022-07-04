## :books: Prepare Dataset
   Please refer to [Pretrained_Dataset](./Pretrained_Dataset.md) and [Downstream Tasks](#bicyclist-downstream-tasks) for further details.
   
   | Tasks | Datasets:point_down: |
   | - | - | 
   | Pre-Training | [ImageNet](https://image-net.org/index.php) <br> [COCO](https://cocodataset.org/#home) |
   | Downstream | [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) <br> [COCO](https://cocodataset.org/#home) |

## :running: Usage - Training
Todo


## :bicyclist: Downstream tasks
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

## Citations
``` bash
Coming Soon
```

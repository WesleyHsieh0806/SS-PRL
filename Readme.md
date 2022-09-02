# Self-Supervised Pyramid Representation Learning <br>for Multi-Label Visual Analysis and Beyond

   This repository provides the official Pytorch implementation of pretraining and downstream evaluations for **SS-PRL**.
   
   [**:paperclip: Paper Link**](https://arxiv.org/abs/2208.14439)
   [**:pencil2: Citations**](#citations)
   
   <div align="center">
  <img width="95%" alt="SS-PRL Gif" src="https://github.com/WesleyHsieh0806/SS-PRL/blob/master/GIF/Framework%20Gif.gif">
   </div>
   
   * **Learning of Patch-Based Pyramid Representation**
   * **Cross-Scale Patch-Level Correlation Learning with Correlation Learners**

---


  <h2> Table of Contents</h2>
  <ul>
    <li>
      <a href="#books-prepare-dataset">Prepare Dataset</a>
      <ul>
        <!-- <li><a href="#built-with">Built With</a></li> -->
      </ul>
    </li>
    <li>
      <a href="#running-usage---training">Usage</a>
    </li>
    <li>
      <a href="#bicyclist-downstream-tasks">Downstream tasks</a>
    </li>
    <li>
      <a href="#citations">Citations</a>
    </li>
  </ul>



---

## :books: Prepare Dataset
   Please refer to [Pretrained_Dataset](./Pretrained_Dataset.md) and [Downstream Tasks](#bicyclist-downstream-tasks) for further details.
   
   | Tasks | Datasets:point_down: |
   | - | - | 
   | Pre-Training | [ImageNet](https://image-net.org/index.php) <br> [COCO](https://cocodataset.org/#home) |
   | Downstream | [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) <br> [COCO](https://cocodataset.org/#home) |

## :running: Usage - Training
### Requirements
- Python 3.6
- [PyTorch](http://pytorch.org) = 1.4.0
- torchvision = 0.5.0
- CUDA 10.1 (Check with nvcc --version)
- [Apex 0.1](https://github.com/NVIDIA/apex) ([Installation](https://github.com/facebookresearch/swav/issues/18#issuecomment-748123838))
- scipy, pandas, numpy
   
``` bash
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

git clone "https://github.com/NVIDIA/apex"
cd apex
git checkout 4a1aa97e31ca87514e17c3cd3bbc03f4204579d0
python setup.py install --cuda_ext
```
### Training with the [shell script](./SS-PRL/train_SSPRL.sh).
   For further details, take a look at the [source file](./SS-PRL/main_SSPRL.py) | [dataset definition](./SS-PRL/src/localpatch_dataset.py) | [utilities](./SS-PRL/src/utils.py)
   ``` bash
   # Training Checklist:
   # 1. modify the DATASET_PATH and EXPERIMENT_PATH in the script
   # 2. BATCH_PER_GPU denotes the batch size per gpu, while --nproc_per_node denotes the number of gpus
   # 3. modify the parameters
   cd SS-PRL
   bash train_SSPRL.sh
   ```


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

   Please Refer to Readme files for [Classification](./benchmarks/classification), [Object-Detection](./benchmarks/detection), and [Semantic Segmentation](https://github.com/WesleyHsieh0806/SS-PRL/tree/master/benchmarks/Segmentation) tasks.

## Citations
``` bash
@misc{hsieh2022selfsupervised,
    title={Self-Supervised Pyramid Representation Learning for Multi-Label Visual Analysis and Beyond},
    author={Cheng-Yen Hsieh and Chih-Jung Chang and Fu-En Yang and Yu-Chiang Frank Wang},
    year={2022},
    eprint={2208.14439},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

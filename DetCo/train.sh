#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee
#PBS -l walltime=24:00:00

############### 1. modify the name of vir-env
#source activate b06901053_swav
#cd $PBS_O_WORKDIR
#module load cuda/cuda-10.0/x86_64

###################### 2. Modify the data path
DATASET_PATH="../COCO/tmp_root"
EXPERIMENT_PATH="./experiments/MLSSL_v2_200ep_bs128"
mkdir -p $EXPERIMENT_PATH
EPOCH=200
BATCH_PERGPU=128

python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 main_MLSSL.py \
--data_path $DATASET_PATH \
--nmb_crops 2 \
--nmb_loc_views 2 \
--size_crops 224 \
--loc_size_crops 255 \
--min_scale_crops 0.14 0.6 \
--max_scale_crops 1. 1. \
--batch_size $BATCH_PERGPU \
--crops_for_assign 0 1 \
--loc_view_for_assign 0 1 \
--epsilon 0.05 \
--feat_dim 128 \
--epochs $EPOCH \
--base_lr 0.6 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--arch resnet50 \
--use_fp16 true \
--grid_perside 3 \
--dump_path $EXPERIMENT_PATH \
--detco-t 0.07 \
--detco-dim 128 \
--detco-k 65536 \
--detco-m 0.999 \
--detco-weightslist 0.1 0.4 0.7 1.0
###


#conda deactivate

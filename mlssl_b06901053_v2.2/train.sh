#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee
#PBS -l walltime=24:00:00

############### 1. modify the name of vir-env
source activate b06901053_swav
cd $PBS_O_WORKDIR
module load cuda/cuda-10.0/x86_64

###################### 2. Modify the data path
DATASET_PATH="../data/train"
EXPERIMENT_PATH="./experiments/MLSSL_v2_200ep_bs256"
Lambda1=0.5
Lambda2=1.0
mkdir -p $EXPERIMENT_PATH
EPOCH=200
BATCH_PERGPU=64

python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 main_MLSSL.py \
--Lambda1 $Lambda1 \
--Lambda2 $Lambda2 \
--data_path $DATASET_PATH \
--nmb_crops 2 6 \
--nmb_loc_views 2 \
--size_crops 224 96 \
--loc_size_crops 255 \
--min_scale_crops 0.14 0.05 0.6 \
--max_scale_crops 1. 0.14 1. \
--batch_size $BATCH_PERGPU \
--crops_for_assign 0 1 \
--loc_view_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_ptypes 3000 \
--nmb_local_ptypes 5000 \
--queue_length 3840 \
--local_queue_length 15000 \
--epoch_queue_introduce 15 \
--epoch_queue_starts 30 \
--epochs $EPOCH \
--base_lr 0.6 \
--final_lr 0.0006 \
--freeze_prototypes_niters 5005 \
--wd 0.000001 \
--warmup_epochs 0 \
--arch resnet50 \
--use_fp16 true \
--grid_perside 3 \
--dump_path $EXPERIMENT_PATH
###


conda deactivate

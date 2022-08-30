# Downstream Object Detection Tasks

## Requirements

We use [detectron2](https://github.com/facebookresearch/detectron2) to train the object detection models. To install the package, please follow the [installation instructions](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).


## Data & Model Preparation

-  Put the [COCO dataset](https://cocodataset.org/#home) under `benchmarks/detection/datasets` directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

-  Convert the pre-trained backbone weights to detectron2's format:
   ```
   cd benchmarks/detection

   WEIGHT_FILE="ss-prl.pth.tar" # pre-trained weight
   OUTPUT_FILE="ss-prl.pkl"     # converted output file

   python convert-pretrain-to-detectron2.py ${WEIGHT_FILE} ${OUTPUT_FILE}
   ```  

## Start training

   ```
   CONVERTED_WEIGHT="ss-prl.pkl" # converted pre-trained weight
   LOG_DIR="./log"               # where to write training logs

   bash run.sh ${CONVERTED_WEIGHT} ${LOG_DIR}
   ```
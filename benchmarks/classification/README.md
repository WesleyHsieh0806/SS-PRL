# Downstream Multi-Label Classification Tasks

## Requirements

In addition to the [requirements](https://github.com/WesleyHsieh0806/SS-PRL#requirements) for SS-PRL pre-training, the following packages are used:
- scikit-learn 

## Fine-Tuning Linear Classifiers

- Fine-tuning on the COCO dataset:
   ```
   DATA="./data/coco"      # path to the coco dataset
   WEIGHT="ss-prl.pth.tar" # pre-trained weight

   bash scropts/linear_coco.sh ${DATA} ${WEIGHT}
   ```
- Fine-tuning on the VOC dataset:
   ```
   DATA="./data/voc"       # path to the voc dataset
   WEIGHT="ss-prl.pth.tar" # pre-trained weight

   bash scropts/linear_voc.sh ${DATA} ${WEIGHT}
   ```

## Semi-Supervised Training

- Semi-Supervised training on the COCO dataset:
   ```
   DATA="./data/coco"      # path to the coco dataset
   WEIGHT="ss-prl.pth.tar" # pre-trained weight
   PERC="1"                # fine-tune with 1 or 10 percent of labels

   bash scropts/semisup_coco.sh ${DATA} ${WEIGHT} ${PERC}
   ```
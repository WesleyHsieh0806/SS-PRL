import os
import argparse

# the dataset format should be COCO/tmproot/label1/XXX.jpg
#                                          /label2/XXX.jpg
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/path/to/COCO/tmproot",
                    help="path to dataset repository")
parser.add_argument("--output_txt", type=str, default="/path/to/COCO/ann.txt",
                    help="path to output annotation txt")

args = parser.parse_args()

with open(args.output_txt, encoding="utf-8", mode="w") as f:
    image_paths = []
    for label in os.listdir(args.data_path):
        if os.path.isdir(os.path.join(args.data_path, label)):
            for image in os.listdir(os.path.join(args.data_path, label)):
                image_paths.append(os.path.join(label, image))
    f.write("\n".join(image_paths))


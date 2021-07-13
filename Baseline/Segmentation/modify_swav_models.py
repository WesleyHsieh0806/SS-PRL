import torch
import os
import argparse

parser = argparse.ArgumentParser(
    description="Modify swav pretrained files into correct format")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained weights")
args = parser.parse_args()

if os.path.isfile(args.pretrained):
    model = torch.load(args.pretrained)
    if "state_dict" in model:
        state_dict = model["state_dict"]

    else:
        state_dict = model

    for k, v in state_dict.items():
        print(k)
    save_dict = {"state_dict", state_dict}

import torch
import os
import argparse

parser = argparse.ArgumentParser(
    description="Modify swav pretrained files into correct format")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained weights")
parser.add_argument("--model", default="", type=str,
                    help="path to pretrained weights")
args = parser.parse_args()

if os.path.isfile(args.pretrained):
    model = torch.load(args.pretrained)
    if "state_dict" in model:
        state_dict = model["state_dict"]

    else:
        state_dict = model
    # Remove module
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    densecl_model = torch.load(args.model)
    for k, v in densecl_model.state_dict().items():
        if k not in list(state_dict):
            print('key "{}" could not be found in provided state dict'.format(k))
        elif state_dict[k].shape != v.shape:
            print(
                'key "{}" is of different shape in model and provided state dict'.format(k))
    for k, v in state_dict.items():
        if k not in list(densecl_model.state_dict()):
            print('Extra key "{}" found!'.format(k))
    save_dict = {"state_dict": state_dict}

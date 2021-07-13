import torch
import os
import argparse

'''
* Since we follow the same setting in semantic segmentation, we have to modify the pretrained weights of SwAV
'''
parser = argparse.ArgumentParser(
    description="Modify swav pretrained files into correct format")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained weights")
parser.add_argument("--model", default="", type=str,
                    help="path of DenseCL pretrained weights")
args = parser.parse_args()

if os.path.isfile(args.pretrained):
    model = torch.load(args.pretrained)
    if "state_dict" in model:
        org_state_dict = model["state_dict"]

    else:
        org_state_dict = model
    # Remove module and projection head
    state_dict = {}
    for k, v in org_state_dict.items():
        if ("projection" not in k) and ("prototype" not in k):
            state_dict[k.replace("module.", "")] = v

    # Check whether the module is the same as DenseCL
    densecl_state = torch.load(args.model)["state_dict"]
    for k, v in densecl_state.items():
        if k not in list(state_dict):
            print('key "{}" could not be found in provided state dict'.format(k))
        elif state_dict[k].shape != v.shape:
            print(
                'key "{}" is of different shape in model and provided state dict'.format(k))
    for k, v in state_dict.items():
        if k not in list(densecl_state):
            print('Extra key "{}" found!'.format(k))
    save_dict = {"state_dict": state_dict}

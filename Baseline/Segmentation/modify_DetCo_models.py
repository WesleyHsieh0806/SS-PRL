import torch
import os
import argparse
import sys
'''
* Since we follow the same setting in semantic segmentation, 
* we have to modify the pretrained weights of SwAV into correct format
'''


def main():

    parser = argparse.ArgumentParser(
        description="Modify swav pretrained files into correct format")
    parser.add_argument("--pretrained", required=True, type=str,
                        help="path to pretrained weights")
    parser.add_argument("--model", default="./Models/DenseCL.pth", type=str,
                        help="path of DenseCL pretrained weights")
    parser.add_argument("--newmodel", required=True, type=str,
                        help="path of Modified Models")
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
            if ("encoder_q" in k) and ("layers" not in k) and ("mlp" not in k):
                state_dict[k.replace("module.encoder_q.", "")] = v

        # Check whether the module is the same as DenseCL
        densecl_state = torch.load(args.model)["state_dict"]
        error = False
        for k, v in densecl_state.items():
            if k not in list(state_dict):
                print('key "{}" could not be found in provided state dict'.format(k))
                error = True
            elif state_dict[k].shape != v.shape:
                print(
                    'key "{}" is of different shape in model and provided state dict'.format(k))
                error = True
        for k, v in state_dict.items():
            if k not in list(densecl_state):
                print('Extra key "{}" found!'.format(k))
                error = True

        if not error:
            # Save the modified model
            print("Save the models at :{}".format(args.newmodel))
            save_dict = {"state_dict": state_dict}
            torch.save(save_dict, args.newmodel)


if __name__ == "__main__":
    main()

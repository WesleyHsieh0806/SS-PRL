##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## NUS School of Computing
## Email: yaoyao.liu@nus.edu.sg
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import os
import numpy as np
import csv
import glob
from shutil import copyfile
from tqdm import tqdm

# argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--imagenet_dir', type=str, help='path to imagenet/train')
parser.add_argument('--output_dir', type=str, help='where to store mini-imagenet')

args = parser.parse_args()

class MiniImageNetGenerator(object):
    def __init__(self, input_args):
        self.input_args = input_args
        if self.input_args.imagenet_dir is not None:
            self.imagenet_dir = self.input_args.imagenet_dir
        else:
            print('You need to specify the ILSVRC2012 source file path')
        self.mini_dir = self.imagenet_dir
        self.processed_img_dir = self.input_args.output_dir

    def process_original_files(self):
        split_lists = ['train', 'val', 'test']
        csv_files = ['./csv_files/train.csv','./csv_files/val.csv', './csv_files/test.csv']

        if not os.path.exists(self.processed_img_dir):
            os.makedirs(self.processed_img_dir)

        for this_split in split_lists:
            filename = './csv_files/' + this_split + '.csv'
            this_split_dir = self.processed_img_dir + '/' + this_split
            if not os.path.exists(this_split_dir):
                os.makedirs(this_split_dir)
            with open(filename) as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                next(csv_reader, None)
                images = {}
                print('Reading IDs....')

                for row in tqdm(csv_reader):
                    if row[1] in images.keys():
                        images[row[1]].append(row[0])
                    else:
                        images[row[1]] = [row[0]]

                print('Writing photos....')
                for cls in tqdm(images.keys()):
                    this_cls_dir = this_split_dir + '/' + cls        
                    if not os.path.exists(this_cls_dir):
                        os.makedirs(this_cls_dir)

                    lst_files = []
                    for file in glob.glob(self.mini_dir + "/*"+cls+"*/*"):
                        lst_files.append(file)

                    lst_index = [int(i[i.rfind('_')+1:i.rfind('.')]) for i in lst_files]
                    index_sorted = sorted(range(len(lst_index)), key=lst_index.__getitem__)

                    index_selected = [int(i[i.index('.') - 4:i.index('.')]) for i in images[cls]]
                    selected_images = np.array(index_sorted)[np.array(index_selected) - 1]
                    for i in np.arange(len(selected_images)):
                        copyfile(lst_files[selected_images[i]],os.path.join(this_cls_dir, images[cls][i]))

if __name__ == "__main__":
    dataset_generator = MiniImageNetGenerator(args)
    dataset_generator.process_original_files()
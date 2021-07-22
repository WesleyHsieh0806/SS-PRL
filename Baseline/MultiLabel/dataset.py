import os
import json
import glob
from collections import defaultdict

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

tr_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    tr_normalize,
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    tr_normalize,
])

class VOCDataset(Dataset):
    def __init__(self, voc_root, is_train=True):
        mode = 'trainval' if is_train else 'test'

        # Find the image sets
        image_set_dir = os.path.join(voc_root, 'ImageSets', 'Main')
        image_sets = glob.glob(os.path.join(image_set_dir, '*_' + mode + '.txt'))
        assert len(image_sets) == 20

        # Read the labels
        self.n_labels = len(image_sets)
        images = defaultdict(lambda:-np.ones(self.n_labels, dtype=np.uint8)) 
        for k, s in enumerate(sorted(image_sets)):
            for l in open(s, 'r'):
                name, lbl = l.strip().split()
                lbl = int(lbl)
                # Switch the ignore label and 0 label (in VOC -1: not present, 0: ignore)
                if lbl < 0:
                    lbl = 0
                elif lbl == 0:
                    lbl = 255
                images[os.path.join(voc_root, 'JPEGImages', name + '.jpg')][k] = lbl
        self.images = [(k, images[k]) for k in images.keys()]
        np.random.shuffle(self.images)
        self.transform = train_transform if is_train else val_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.transform(Image.open(self.images[i][0]).convert('RGB'))
        label = torch.tensor(self.images[i][1]).float()
        return image, label
    def _parse_voc_xml(self, node):
        label = torch.zeros(len(self.cat_map))
        for child in node:
            if child.tag == 'object':
                label[self.cat_map[list(child)[0].text]] = 1.
        return label

class COCODataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        mode = 'train' if is_train else 'val'

        # load annotation file
        annotation_file = os.path.join(data_dir, 'annotations', f'instances_{mode}2014.json')
        with open(annotation_file, 'r') as f:
            annotation = json.load(f)

        # construct image paths
        id_map = {}
        self.image_paths = []
        image_dir = os.path.join(data_dir, f'{mode}2014')
        for idx, image in enumerate(annotation['images']):
            id_map[image['id']] = idx
            self.image_paths.append(os.path.join(image_dir, image['file_name']))

        # create category mapping
        cat_map = {}
        for idx, cat in enumerate(annotation['categories']):
            cat_map[cat['id']] = idx
        
        # construct labels
        self.labels = torch.zeros(len(annotation['images']), len(cat_map))
        for instance in annotation['annotations']:
            img_idx = id_map[instance['image_id']]
            cat_idx = cat_map[instance['category_id']]
            self.labels[img_idx, cat_idx] = 1.

        # set transform
        self.transform = train_transform if is_train else val_transform
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_path = self.image_paths[idx]
        try:
            image = self.transform(Image.open(image_path).convert('RGB'))
        except:
            image = None
            print(image_path)
        return image, label

if __name__ == '__main__':
    dataset = COCODataset('../COCO', True)
    print('start')
    for i in range(len(dataset)):
        print(f'\r{i}/{len(dataset)}', end='')
        image, label = dataset[i]

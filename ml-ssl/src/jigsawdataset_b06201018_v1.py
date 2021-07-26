# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

logger = getLogger()


class JigsawDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
    ):
        super(JigsawDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        randomresizedcrop = transforms.RandomResizedCrop(
            size_crops[0],
            scale=(min_scale_crops[0], max_scale_crops[0]),
        )
        trans.extend([transforms.Compose([
            randomresizedcrop,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Compose(color_transform),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        ] * nmb_crops[0])
        self.global_trans = trans

        self.local_trans = JigsawCrop(num_copies=nmb_crops[0])

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        global_crops = list(map(lambda trans: trans(image), self.global_trans))
        local_crops = self.local_trans(image)
        if self.return_index:
            return index, global_crops, local_crops
        return  global_crops, local_crops


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class JigsawCrop(object):
    def __init__(self, num_copies=1):
        self.num_copies = num_copies
        self.crop_areas = [(i*85, j*85, (i+1)*85, (j+1)*85) for j in range(3) for i in range(3)]
        
        self.randomresizedcrop = transforms.RandomResizedCrop(255, scale=(0.6, 1.0))
        
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(p=0.5),
            get_color_distortion(),
            PILRandomGaussianBlur(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, pic):
        pic = self.randomresizedcrop(pic)
        crops = [pic.crop(crop_area) for crop_area in self.crop_areas]
        jigsaw_copies = []
        for i in range(self.num_copies):
            jigsaw_copies.extend(
                list(map(lambda crop: self.transform(crop), crops))
            )
        return jigsaw_copies
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
import torch

logger = getLogger()


class JigsawDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_path,
        size_crops,
        loc_size_crops,
        nmb_crops,
        nmb_loc_views,
        min_scale_crops,
        max_scale_crops,
        grid_perside,
        size_dataset=-1,
        return_index=False,
    ):
        # The ImageFolder Dataset reads from the data path and deals with the index of each sample automatically
        # The folder has to be structured like
        # root - dog - 1.png
        #              2.png
        # root - cat - 1.png
        super(JigsawDataset, self).__init__(data_path)

        # size_crops:[224, 96] loc_size_crops:[255]
        # Error checking
        assert len(size_crops) == 2
        assert len(loc_size_crops) == 1
        assert len(size_crops) == len(nmb_crops)
        assert len(loc_size_crops) == len(nmb_loc_views)
        assert len(min_scale_crops) == (len(nmb_crops) + len(nmb_loc_views))
        assert len(max_scale_crops) == (len(nmb_crops) + len(nmb_loc_views))
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []

        # The global augmentation
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

        # Resize before slicing images into local patches
        self.rsbf_local = transforms.Compose([
            transforms.RandomResizedCrop(
                loc_size_crops[-1],
                scale=(min_scale_crops[-1], max_scale_crops[-1]),
            ),
            transforms.ToTensor()
        ])
        # The augmentation for each local patch
        self.local_tran = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Compose(color_transform),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        # Record number of local images
        self.nmb_limg = nmb_loc_views[0]
        self.grid_perside = grid_perside

    def _get_localpatch(self, img, view_list, random_order=True):
        '''
        * Append all local patches into view_list
        * view_list:[Global image1, Global image2, ....]
        * -> [Gimg1, Gimg2, ..., LocalPatc1(position 0), ..., LocalPatc2(position 0), LocalPatc2(position 8)]
        '''
        local_img = self.rsbf_local(img)

        # To permute the order for local patched, we use torch.randperm here
        x_order = torch.randperm(
            self.grid_perside) if random_order else range(self.grid_perside)
        y_order = torch.randperm(
            self.grid_perside) if random_order else range(self.grid_perside)

        grid_size = local_img.shape[-1] // self.grid_perside
        for idx in range(self.nmb_limg):
            for i in x_order:
                for j in y_order:
                    x_offset = i * grid_size
                    y_offset = j * grid_size
                    local_patch = local_img[
                        :, y_offset: y_offset + grid_size, x_offset: x_offset + grid_size
                    ]
                    # Append this local patch
                    view_list.append(self.local_tran(local_patch))

    def __getitem__(self, index) -> list:
        path, _ = self.samples[index]
        image = self.loader(path)
        view_list = list(map(lambda trans: trans(image), self.trans))

        # Append the local patches into view_list
        self._get_localpatch(image, view_list)
        if self.return_index:
            return index, view_list
        # now view_list contains 2views, 6m-crops and 3x3x2 patches
        return view_list


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

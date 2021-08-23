###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2018
###########################################################################

import os
import sys
import random
import numpy as np
from tqdm import tqdm, trange
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform

from .base import BaseDataset

from os import listdir
from os.path import isfile, join
from scipy.interpolate import interp2d


class BinaryImageDataset(BaseDataset):
    def __init__(self, root_pos=os.path.expanduser('/BS/work/data_pos'), root_neg=os.path.expanduser('/BS/work/data_neg'), flip=True, **kwargs):
        super(BinaryImageDataset, self).__init__(
            root_pos, root_neg, flip,  **kwargs)
        self.files = get_data_pairs(self.root_pos, self.root_neg)
        assert (len(self.files[0]) == len(self.files[1]))
        if len(self.files) == 3:
            assert (len(self.files[0]) == len(self.files[2]))
        if len(self.files[0]) == 0:
            raise RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n")
        print("Found %d examples" % len(self.files[0]))

    def __getitem__(self, index):
        tmp = Image.open(self.files[0][index][:])
        data = np.array(tmp)

        data = data.transpose(2, 0, 1)
            
        if self.flip:
            flip_step = np.random.randint(0, 2) * 2 - 1
            data = data[:, :, ::flip_step]

        label = self.files[1][index]

        data = torch.from_numpy(data.copy()).float()
        label = torch.tensor(label).long()

        return data, label, self.files[0][index]

    def __len__(self):
        return len(self.files[0])



def get_data_pairs(pos_folder, neg_folder):
    def get_pairs(pos_folder, neg_folder):
        pos_data = sorted([os.path.join(pos_folder, f) for f in listdir(pos_folder) if isfile(join(pos_folder, f))])
        neg_data = sorted([os.path.join(neg_folder, f) for f in listdir(neg_folder) if isfile(join(neg_folder, f))])
        return pos_data, neg_data

    pos_data, neg_data = get_pairs(pos_folder, neg_folder)
    return [pos_data+neg_data, [1]*len(pos_data)+[0]*len(neg_data)]
    


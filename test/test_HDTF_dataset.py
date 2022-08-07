'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-07 23:55:58
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import sys
sys.path.append('./')
sys.path.append('../')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import torchvision
import numpy as np
from easydict import EasyDict


def test_HDTFDataset():
    from data.HDTF_dataset import HDTFDataset

    opt = EasyDict()
    opt.data_root = "./dataset/HDTF_face3dmmformer"
    opt.semantic_radius = 13
    opt.split = "./dataset/train_HDTF_face3dmmformer.txt"
    opt.statistics_file = "./dataset/HDTF_face3dmmformer_statistics.txt"

    dataset = HDTFDataset(opt)
    print(len(dataset))

    item = dataset[0]


if __name__ == "__main__":
    test_HDTFDataset()
    print("Done")
    pass

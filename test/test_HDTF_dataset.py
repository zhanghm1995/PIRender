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
import torchvision


def test_HDTFDataset():
    from data.HDTF_dataset import HDTFDataset

    opt = EasyDict()
    opt.data_root = "./dataset/HDTF_face3dmmformer"
    opt.semantic_radius = 13
    opt.split = "./dataset/train_HDTF_face3dmmformer.txt"
    opt.statistics_file = "./dataset/HDTF_face3dmmformer_statistics.txt"

    dataset = HDTFDataset(opt)
    print(len(dataset))

    element = dataset[2910]

    vis_images = []
    for key, value in element.items():
        print(key, value.shape)
        if "image" in key:
            value = (value + 1.0) / 2.0
            vis_images.append(value)

    torchvision.utils.save_image(vis_images, "hdtf_test.jpg", padding=0)
    

def test_HDTFVideoDataset():
    from data.HDTF_video_dataset import HDTFVideoDataset

    opt = EasyDict()
    opt.data_root = "./dataset/HDTF_face3dmmformer"
    opt.semantic_radius = 13
    opt.split = "./dataset/train_HDTF_face3dmmformer.txt"
    opt.statistics_file = "./dataset/HDTF_face3dmmformer_statistics.txt"

    dataset = HDTFVideoDataset(opt, is_inference=True)
    print(len(dataset))

    element = dataset.load_next_video()
    print(element.keys())

    print(element['source_semantics'].shape)


if __name__ == "__main__":
    test_HDTFVideoDataset()
    print("Done")

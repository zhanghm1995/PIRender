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
import time
from tqdm import tqdm


def profile_HDTFDataset():
    from data.HDTF_dataset import HDTFDataset
    print(f"profile_HDTFDataset")

    opt = EasyDict()
    opt.data_root = "./dataset/HDTF_face3dmmformer"
    opt.semantic_radius = 13
    opt.split = "./dataset/train_HDTF_face3dmmformer.txt"
    opt.statistics_file = "./dataset/HDTF_face3dmmformer_statistics.txt"

    dataset = HDTFDataset(opt)
    print(len(dataset))

    start = time.time()
    for i in tqdm(range(10)):
        element = dataset[i]
        print(i)
    end = time.time()
    print(f"elapsed time is {end - start}, and mean time is {(end - start) / 10}")


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


def test_HDTFDemoDataset():
    from data.HDTF_demo_dataset import HDTFDemoDataset
    config = EasyDict()
    config.data_root = "./dataset/HDTF_face3dmmformer/val"
    config.video_name = "WRA_KellyAyotte_000"
    config.pred_dir = "/home/zhanghm/Research/V100/TalkingFaceFormer/test_dir/demo_audio_sing_song_PPE"

    data = HDTFDemoDataset(**config)
    print(len(data))

    element = data[0]

    vis_images = []
    for key, value in element.items():
        print(key, value.shape)
        if "image" in key:
            value = (value + 1.0) / 2.0
            vis_images.append(value)

    torchvision.utils.save_image(vis_images, "demo_hdtf_test.jpg", padding=0)


if __name__ == "__main__":
    profile_HDTFDataset()
    print("Done")

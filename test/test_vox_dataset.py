'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-09-12 10:54:54
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the offical dataset class to load Vox1 dataset.
'''

import sys
sys.path.append("./")
sys.path.append("../")

import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import time
import numpy as np
import torchvision
import cv2
from PIL import Image
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


denormalize_transform = transforms.Normalize((-1, -1, -1), (2, 2, 2), inplace=True)


def test_VoxDataset():
    from data.vox_dataset import VoxDataset

    config = OmegaConf.load("./config/face.yaml")

    dataset = VoxDataset(config.data, is_inference=False)
    print("The length of the dataset is: ", len(dataset))

    person_ids = dataset.person_ids
    print(len(person_ids), person_ids[:3])
    print(len(set(person_ids)))

    entry = dataset[10]
    for key, value in entry.items():
        print(key, value.shape)
    

    save_image = torch.stack((entry['source_image'], entry['target_image']), dim=0)
    save_image = denormalize_transform(save_image)
    torchvision.utils.save_image(save_image, "test_vox.png", padding=0)

    print(dataset.idx_by_person_id['id10499'])


def test_VoxPngDataset():
    from data.vox_png_dataset import VoxPngDataset
    config = OmegaConf.load("./config/face_vox_png.yaml")

    dataset = VoxPngDataset(config.data, is_inference=False)
    print("The length of the dataset is: ", len(dataset))

    print(len(dataset.video_items))
    print(dataset.idx_by_person_id['id10499'])

    entry = dataset[10]
    save_image = torch.stack((entry['source_image'], entry['target_image']), dim=0)
    save_image = denormalize_transform(save_image)
    torchvision.utils.save_image(save_image, "test_vox.png", padding=0)


if __name__ == "__main__":
    test_VoxPngDataset()

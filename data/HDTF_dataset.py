'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-07 22:47:08
Email: haimingzhang@link.cuhk.edu.cn
Description: The dataset class for loading HDTF dataset.
'''

import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import time
import numpy as np
import random

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from collections import defaultdict


class HDTFDataset(Dataset):
    def __init__(self, opt) -> None:
        super().__init__()

        self.data_root = opt.data_root
        self.semantic_radius = opt.semantic_radius

        if "train" in opt.split:
            self.data_root = osp.join(self.data_root, "train")

        ## Get all key frames
        self.file_paths = open(opt.split).read().splitlines()

        self.video_name_to_imgs_list_dict = defaultdict(list)
        for line in self.file_paths:
            name = line.split('/')[0]
            self.video_name_to_imgs_list_dict[name].append(line)

        ## Read the frames number statistics information
        lines = open(opt.statistics_file).read().splitlines()
        self.video_frame_length_dict = {}
        for line in lines:
            entry = line.strip().split(' ')
            self.video_frame_length_dict[entry[0]] = int(entry[1])

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index: int):
        data = {}

        ## Get the video name
        file_path = self.file_paths[index]

        img_dir, file_name = osp.split(file_path)
        video_name = osp.dirname(img_dir)

        ## Read the source image and source semantics
        source_img_path = osp.join(self.data_root, file_path + ".jpg")
        source_image = Image.open(source_img_path).convert("RGB")
        data['source_image'] = self.transform(source_image)


        ## Read arbitrary reference image
        video_length = self.video_frame_length_dict[video_name] ## TODO

        ref_img_idx = random.choice(range(video_length))
        reference_image = Image.open(osp.join(self.data_root, img_dir, f"{ref_img_idx:06d}.jpg")).convert("RGB")
        data['reference_image'] = self.transform(reference_image)

        ## Get the 3DMM rendered face image
        index_seq_window = self.get_index_seq_window(int(file_name), video_length)
        print(index_seq_window)

        return data

    def _build_dataset(self):
        self.total_frames_list = []
        self.length_token_list = [] # increamental length list
        self.all_videos_dir = []

        total_length = 0
        for video_name, all_imgs_list in self.video_name_to_imgs_list_dict.items():
            self.all_videos_dir.append(video_name)

            num_frames = len(all_imgs_list)
            self.total_frames_list.append(num_frames)

            total_length += num_frames
            self.length_token_list.append(total_length)

    def _get_data(self, index):
        """Get the seperate index location from the total index

        Args:
            index (int): index in all avaible sequeneces
        
        Returns:
            main_idx (int): index specifying which video
            sub_idx (int): index specifying what the start index in this sliced video
        """
        def fetch_data(length_list, index):
            assert index < length_list[-1]
            temp_idx = np.array(length_list) > index
            list_idx = np.where(temp_idx==True)[0][0]
            sub_idx = index
            if list_idx != 0:
                sub_idx = index - length_list[list_idx - 1]
            return list_idx, sub_idx

        main_idx, sub_idx = fetch_data(self.length_token_list, index)
        return main_idx, sub_idx

    def get_index_seq_window(self, index, num_frames):
        seq = list(range(index - self.semantic_radius, index + self.semantic_radius + 1))
        seq = [ min(max(item, 0), num_frames-1) for item in seq ]
        return seq
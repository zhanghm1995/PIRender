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
from scipy.io import loadmat

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from collections import defaultdict

from visualizer.render_utils import MyMeshRender
from .face_utils import get_coeff_vector, rescale_mask_V2, get_contour


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
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])
        
        self.face_renderer = MyMeshRender()
        
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

        coeff_3dmm_list, semantic_params_list = [], []
        for i in index_seq_window:
            mat_fp = osp.join(self.data_root, video_name, "deep3dface", f"{i:06d}.mat")
            semantic_params, coeff_3dmm_all = self.read_face3dmm_params(mat_fp)
            coeff_3dmm_list.append(coeff_3dmm_all)
            semantic_params_list.append(semantic_params)

        coeff_3dmm_all_tensor = torch.Tensor(np.concatenate(coeff_3dmm_list, axis=0)).permute(1,0)
        source_semantics = torch.Tensor(np.concatenate(semantic_params_list, axis=0)).permute(1,0)
        data['source_semantics'] = source_semantics

        ## Get the rendered face image
        curr_semantics = coeff_3dmm_all_tensor[:, 13:14].permute(1, 0) # (1, 260)
        curr_face3dmm_params = curr_semantics[:, :257] # (1, 257)
        curr_trans_params = curr_semantics[:, -3:]

        rendered_image_numpy, mask = self.face_renderer.compute_rendered_face(curr_face3dmm_params, None)
        # get the rescaled_rendered_image (256, 256, 3)
        rescaled_rendered_image = rescale_mask_V2(rendered_image_numpy[0], curr_trans_params[0], original_shape=(512, 512))
        data['rendered_image'] = self.transform(rescaled_rendered_image)

        ## get the rescaled mask image
        mask = mask.permute(0, 2, 3, 1).cpu().numpy() * 255 # (B, 224, 224, 1)
        mask = mask.astype(np.uint8)
        rescaled_mask_image = rescale_mask_V2(mask[0], curr_trans_params[0], original_shape=(256, 256))
        rescaled_mask_image = get_contour(np.array(rescaled_mask_image)[..., 0].astype(np.uint8))
        rendered_face_mask_img_tensor = torch.FloatTensor(np.array(rescaled_mask_image)) / 255.0
        rendered_face_mask_img_tensor = rendered_face_mask_img_tensor[..., None].permute(2, 0, 1) # (1, H, W)
        
        ## Get the blended face image
        blended_img_tensor = data['source_image'] * (1 - rendered_face_mask_img_tensor) + \
                             data['rendered_image'] * rendered_face_mask_img_tensor
        
        data['blended_image'] = blended_img_tensor

        return data

    def read_face3dmm_params(self, file_path):
        file_mat = loadmat(file_path)
        coeff_3dmm = get_coeff_vector(file_mat)
        crop_param = file_mat['transform_params']
        _, _, ratio, t0, t1 = np.hsplit(crop_param.astype(np.float32), 5)
        crop_param = np.concatenate([ratio, t0, t1], 1)
        coeff_3dmm_all = np.concatenate([coeff_3dmm, crop_param], 1)

        ## get the semantic params
        semantic_params = get_coeff_vector(file_mat, key_list=['exp', 'angle', 'trans'])
        semantic_params = np.concatenate([semantic_params, crop_param], 1)
        return semantic_params, coeff_3dmm_all

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
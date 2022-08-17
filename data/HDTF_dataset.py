'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-07 22:47:08
Email: haimingzhang@link.cuhk.edu.cn
Description: The dataset class for loading HDTF dataset for training.
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
from torch import Tensor
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from collections import defaultdict

from visualizer.render_utils import MyMeshRender
from .face_utils import get_coeff_vector, rescale_mask_V2, get_contour


class HDTFDataset(Dataset):
    def __init__(self, opt, is_inference=None) -> None:
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
        
        self.face_renderer = MyMeshRender()

        self.half_mask = np.zeros((224, 224, 1), dtype=np.uint8)
        self.half_mask[:128, ...] = 255

        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 31))

        self.blend_image_ablation = True
        
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
        video_length = self.video_frame_length_dict[video_name]

        ref_img_idx = random.choice(range(video_length))
        reference_image = Image.open(osp.join(self.data_root, img_dir, f"{ref_img_idx:06d}.jpg")).convert("RGB")
        data['reference_image'] = self.transform(reference_image)

        ## Get the 3DMM rendered face image
        source_mat_fp = osp.join(self.data_root, video_name, "deep3dface", f"{file_name}.mat")
        source_semantic_params, source_coeff_3dmm_all = self.read_face3dmm_params(source_mat_fp)

        ## Get the rendered face image
        curr_semantics = torch.from_numpy(source_coeff_3dmm_all)
        curr_face3dmm_params = curr_semantics[:, :257] # (1, 257)
        curr_trans_params = curr_semantics[:, -3:]

        rendered_image_numpy, mask = self.face_renderer.compute_rendered_face(curr_face3dmm_params, None)
        # get the rescaled_rendered_image (256, 256, 3)
        rescaled_rendered_image = rescale_mask_V2(
            rendered_image_numpy[0], curr_trans_params[0], original_shape=(512, 512))
        data['rendered_image'] = self.transform(rescaled_rendered_image)

        if not self.blend_image_ablation:
            ## get the rescaled mask image
            rendered_face_mask_img_tensor = self.get_rescaled_mask(mask, curr_trans_params, mask_augment=True)
            
            ## Get the blended face image
            blended_img_tensor = data['source_image'] * (1 - rendered_face_mask_img_tensor) + \
                                data['rendered_image'] * rendered_face_mask_img_tensor
        else:
            blended_img_tensor = data['rendered_image']

        data['blended_image'] = blended_img_tensor

        return data

    def __getitem_with_AdaIN__(self, index: int):
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
        rescaled_rendered_image = rescale_mask_V2(
            rendered_image_numpy[0], curr_trans_params[0], original_shape=(512, 512))
        data['rendered_image'] = self.transform(rescaled_rendered_image)

        ## get the rescaled mask image
        rendered_face_mask_img_tensor = self.get_rescaled_mask(mask, curr_trans_params, mask_augment=True)
        
        ## Get the blended face image
        blended_img_tensor = data['source_image'] * (1 - rendered_face_mask_img_tensor) + \
                             data['rendered_image'] * rendered_face_mask_img_tensor
        
        data['blended_image'] = blended_img_tensor

        return data

    def get_rescaled_mask(self, mask: Tensor, trans_params, mask_augment=False):
        mask = mask.permute(0, 2, 3, 1).cpu().numpy() * 255 # (B, 224, 224, 1)
        mask = mask.astype(np.uint8)
        
        if mask_augment:
            ## 1) Remove the mouth hole
            closing = cv2.morphologyEx(mask[0], cv2.MORPH_CLOSE, self.closing_kernel)

            ## 2) Dilate the mask
            img_dialate = cv2.dilate(closing, kernel=self.dilate_kernel, iterations=1)

            img_dialate_half = cv2.bitwise_and(img_dialate, 255 - self.half_mask)
            closing_half = cv2.bitwise_and(closing, self.half_mask)
            combined_mask = cv2.bitwise_or(img_dialate_half, closing_half)
            
            rescaled_mask_image = rescale_mask_V2(combined_mask[..., None], trans_params[0], original_shape=(512, 512))
            rescaled_mask_image = np.array(rescaled_mask_image)[..., 0] # to (512, 512)
        else:
            rescaled_mask_image = rescale_mask_V2(mask[0], trans_params[0], original_shape=(512, 512))
            rescaled_mask_image = get_contour(np.array(rescaled_mask_image)[..., 0].astype(np.uint8))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            rescaled_mask_image = cv2.erode(rescaled_mask_image, kernel=kernel, iterations=1)

        rendered_face_mask_img_tensor = torch.FloatTensor(np.array(rescaled_mask_image)) / 255.0
        rendered_face_mask_img_tensor = rendered_face_mask_img_tensor[..., None].permute(2, 0, 1) # (1, H, W)
        return rendered_face_mask_img_tensor

    def read_face3dmm_params(self, file_path):
        file_mat = loadmat(file_path)
        coeff_3dmm = get_coeff_vector(file_mat)
        crop_param = file_mat['transform_params']
        _, _, ratio, t0, t1 = np.hsplit(crop_param.astype(np.float32), 5)
        crop_param = np.concatenate([ratio, t0, t1], 1)
        coeff_3dmm_all = np.concatenate([coeff_3dmm, crop_param], 1) # (1, 260)

        ## get the semantic params
        semantic_params = get_coeff_vector(file_mat, key_list=['exp', 'angle', 'trans'])
        semantic_params = np.concatenate([semantic_params, crop_param], 1) # (1, 73)
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

    def get_index_seq_window(self, index, num_frames):
        seq = list(range(index - self.semantic_radius, index + self.semantic_radius + 1))
        seq = [ min(max(item, 0), num_frames-1) for item in seq ]
        return seq
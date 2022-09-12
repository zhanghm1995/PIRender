'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-09-12 11:40:46
Email: haimingzhang@link.cuhk.edu.cn
Description: Load the png format Vox1 dataset.
'''

import os
import os.path as osp
import lmdb
import random
import collections
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class VoxPngDataset(Dataset):
    def __init__(self, opt, is_inference):
        self.semantic_radius = opt.semantic_radius

        self.data_root = osp.join(opt.path, "train") \
            if not is_inference else osp.join(opt.path, "test")

        self.video_items, self.person_ids, self.idx_by_person_id, self.video_frame_length_dict = \
            self._build_dataset(self.data_root)

        self.person_ids = self.person_ids * 100

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])

    def _build_dataset(self, data_dir):
        all_videos_name = sorted(os.listdir(data_dir))
        person_ids = sorted(list({video.split('#')[0] for video in all_videos_name}))

        idx_by_persion_id_dict = collections.defaultdict(list)
        for index, video_item in enumerate(all_videos_name):
            person_id = video_item.split('#')[0]
            idx_by_persion_id_dict[person_id].append(index)
        
        ## Get the number of frames in each video
        video_frame_length_dict = {}
        for video_name in all_videos_name:
            meta_info_path = osp.join(data_dir, video_name, 'meta_info.yaml')
            meta_info = OmegaConf.load(meta_info_path)
            video_frame_length_dict[video_name] = int(meta_info.num_frames)

        return all_videos_name, person_ids, idx_by_persion_id_dict, video_frame_length_dict

    def __getitem__(self, index):
        data = {}

        person_id = self.person_ids[index]
        video_item = self.video_items[random.choices(self.idx_by_person_id[person_id], k=1)[0]]
        frame_source, frame_target = self.random_select_frames(video_item)
        print(frame_source, frame_target)

        video_images_dir = osp.join(self.data_root, video_item, "face_image")

        source_image_path = osp.join(video_images_dir, f"{frame_source:06d}.png")
        img1 = Image.open(source_image_path).convert("RGB")
        data['source_image'] = self.transform(img1)
        # data['source_semantics'], coeff_3dmm_all = self.transform_semantic(semantics_numpy, frame_source)

        target_image_path = osp.join(video_images_dir, f"{frame_target:06d}.png")
        img2 = Image.open(target_image_path).convert("RGB")
        data['target_image'] = self.transform(img2)

        return data

    def group_by_key(self, video_list, key):
        return_dict = collections.defaultdict(list)
        for index, video_item in enumerate(video_list):
            return_dict[video_item[key]].append(index)
        return return_dict  
    
    def __len__(self):
        return len(self.person_ids)

    def random_select_frames(self, video_item):
        num_frame = self.video_frame_length_dict[video_item]
        frame_idx = random.choices(list(range(num_frame)), k=2)
        return frame_idx[0], frame_idx[1]

    def transform_semantic(self, semantic, frame_index):
        index = self.obtain_seq_index(frame_index, semantic.shape[0])
        coeff_3dmm = semantic[index,...]
        id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:,80:144] #expression
        tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:,224:227] #euler angles for pose
        gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:,254:257] #translation
        crop = coeff_3dmm[:,257:260] #crop param

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        coeff_3dmm_complete = np.concatenate([id_coeff, ex_coeff, tex_coeff, angles, gamma, translation, crop], 1)

        return torch.Tensor(coeff_3dmm).permute(1,0), torch.Tensor(coeff_3dmm_complete).permute(1,0)

    def obtain_seq_index(self, index, num_frames):
        seq = list(range(index-self.semantic_radius, index+self.semantic_radius+1))
        seq = [ min(max(item, 0), num_frames-1) for item in seq ]
        return seq

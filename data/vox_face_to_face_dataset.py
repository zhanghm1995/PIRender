'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-06 19:28:38
Email: haimingzhang@link.cuhk.edu.cn
Description: The face to face translation dataset for VoxCeleb2
'''

import os
import lmdb
import random
import collections
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from visualizer.render_utils import MyMeshRender
from .face_utils import get_masked_region, rescale_mask_V2, get_contour


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


class VoxFace2FaceDataset(Dataset):
    def __init__(self, opt, is_inference):
        path = opt.path
        self.env = lmdb.open(
            os.path.join(path, str(opt.resolution)),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        list_file = "test_list.txt" if is_inference else "train_list.txt"
        list_file = os.path.join(path, list_file)
        with open(list_file, 'r') as f:
            lines = f.readlines()
            videos = [line.replace('\n', '') for line in lines]

        self.resolution = opt.resolution
        self.semantic_radius = opt.semantic_radius
        self.video_items, self.person_ids = self.get_video_index(videos)
        self.idx_by_person_id = self.group_by_key(self.video_items, key='person_id')
        self.person_ids = self.person_ids * 100

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])

        self.face_renderer = MyMeshRender()

    def get_video_index(self, videos):
        video_items = []
        for video in videos:
            video_items.append(self.Video_Item(video))

        person_ids = sorted(list({video.split('#')[0] for video in videos}))

        return video_items, person_ids            

    def group_by_key(self, video_list, key):
        return_dict = collections.defaultdict(list)
        for index, video_item in enumerate(video_list):
            return_dict[video_item[key]].append(index)
        return return_dict  
    
    def Video_Item(self, video_name):
        video_item = {}
        video_item['video_name'] = video_name
        video_item['person_id'] = video_name.split('#')[0]
        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], 'length')
            length = int(txn.get(key).decode('utf-8'))
        video_item['num_frame'] = length
        
        return video_item

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, index):
        data = {}

        person_id = self.person_ids[index]
        video_item = self.video_items[random.choices(self.idx_by_person_id[person_id], k=1)[0]]
        frame_source, frame_ref = self.random_select_frames(video_item)

        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(video_item['video_name'], frame_source)
            img_bytes_1 = txn.get(key) 
            key = format_for_lmdb(video_item['video_name'], frame_ref)
            img_bytes_2 = txn.get(key) 

            semantics_key = format_for_lmdb(video_item['video_name'], 'coeff_3dmm')
            semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
            semantics_numpy = semantics_numpy.reshape((video_item['num_frame'],-1))
        
        img1 = Image.open(BytesIO(img_bytes_1))
        data['source_image'] = self.transform(img1)
        data['source_semantics'], coeff_3dmm_all = self.transform_semantic(semantics_numpy, frame_source)

        img2 = Image.open(BytesIO(img_bytes_2))
        data['reference_image'] = self.transform(img2)

        ## Get the rendered face image
        curr_semantics = coeff_3dmm_all[:, 13:14].permute(1, 0) # (1, 260)
        curr_face3dmm_params = curr_semantics[:, :257] # (1, 257)
        curr_trans_params = curr_semantics[:, -3:]

        rendered_image, mask = self.face_renderer.compute_rendered_face(curr_face3dmm_params, None)
        # get the rescaled_rendered_image (256, 256, 3)
        rescaled_rendered_image = rescale_mask_V2(rendered_image[0], curr_trans_params[0], original_shape=(256, 256))
        data['rendered_image'] = self.transform(rescaled_rendered_image)

        ## get the rescaled mask image
        mask = mask.permute(0, 2, 3, 1).cpu().numpy() * 255 # (B, 224, 224, 1)
        mask = mask.astype(np.uint8)
        rescaled_mask_image = rescale_mask_V2(mask[0], curr_trans_params[0], original_shape=(256, 256))
        rescaled_mask_image = get_contour(np.array(rescaled_mask_image)[..., 0].astype(np.uint8))
        rendered_face_mask_img_tensor = torch.FloatTensor(np.array(rescaled_mask_image)) / 255.0
        rendered_face_mask_img_tensor = rendered_face_mask_img_tensor[..., None].permute(2, 0, 1) # (1, H, W)

        ## Get the binary mask image from the 3DMM rendered face image
        # rendered_face_mask_img = get_masked_region(np.array(rescaled_rendered_image))[..., None]
        # rendered_face_mask_img_tensor = torch.FloatTensor(rendered_face_mask_img) / 255.0
        # rendered_face_mask_img_tensor = rendered_face_mask_img_tensor.permute(2, 0, 1) # (1, H, W)
        
        ## Get the blended face image
        blended_img_tensor = data['source_image'] * (1 - rendered_face_mask_img_tensor) + \
                             data['rendered_image'] * rendered_face_mask_img_tensor
        
        data['blended_image'] = blended_img_tensor
        return data
    
    def random_select_frames(self, video_item):
        num_frame = video_item['num_frame']
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


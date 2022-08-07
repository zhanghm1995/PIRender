'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-07 14:13:30
Email: haimingzhang@link.cuhk.edu.cn
Description: The dataset class for face to face inference pipeline.
'''

import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import torch
import torchvision.transforms as transforms

from data.vox_dataset import format_for_lmdb
from data.vox_face_to_face_dataset import VoxFace2FaceDataset
from .face_utils import get_masked_region, rescale_mask_V2, get_contour


class VoxFace2FaceVideoDataset(VoxFace2FaceDataset):
    def __init__(self, opt, is_inference):
        super(VoxFace2FaceVideoDataset, self).__init__(opt, is_inference)
        self.video_index = -1
        self.cross_id = False
        # whether normalize the crop parameters when performing cross_id reenactments
        # set it as "True" always brings better performance
        self.norm_crop_param = True

    def __len__(self):
        return len(self.video_items)

    def load_next_video(self):
        self.video_index += 1
        
        video_item = self.video_items[self.video_index]
        source_video_item = video_item

        num_frames = video_item['num_frame']

        data = {}
        data['source_image'], data['source_semantics'] = [], []
        data['blended_image'], data['reference_image'] = [], []
        data['masked_image'], data['rendered_image'] = [], []
        with self.env.begin(write=False) as txn:
            semantics_key = format_for_lmdb(video_item['video_name'], 'coeff_3dmm')
            semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
            semantics_numpy = semantics_numpy.reshape((video_item['num_frame'],-1))

            for frame_index in range(num_frames):
                key = format_for_lmdb(video_item['video_name'], frame_index)
                img_bytes_1 = txn.get(key) 
                img1 = Image.open(BytesIO(img_bytes_1))
                source_image = self.transform(img1)
                data['source_image'].append(source_image)

                reference_frame_idx = random.choice(list(range(num_frames)))
                key = format_for_lmdb(video_item['video_name'], reference_frame_idx)
                img_bytes_2 = txn.get(key)
                img2 = Image.open(BytesIO(img_bytes_2))
                ref_image = self.transform(img2)
                data['reference_image'].append(ref_image)

                source_semantics, coeff_3dmm_all = self.transform_semantic(
                    semantics_numpy, frame_index, crop_norm_ratio=None)

                data['source_semantics'].append(source_semantics)

                ## Get the rendered face image
                curr_semantics = coeff_3dmm_all[:, 13:14].permute(1, 0) # (1, 260)
                curr_face3dmm_params = curr_semantics[:, :257] # (1, 257)
                curr_trans_params = curr_semantics[:, -3:]

                rendered_image, mask = self.face_renderer.compute_rendered_face(curr_face3dmm_params, None)
                # get the rescaled_rendered_image (256, 256, 3)
                rescaled_rendered_image = rescale_mask_V2(rendered_image[0], curr_trans_params[0], original_shape=(256, 256))
                rendered_image = self.transform(rescaled_rendered_image)

                ## get the rescaled mask image
                mask = mask.permute(0, 2, 3, 1).cpu().numpy() * 255 # (B, 224, 224, 1)
                mask = mask.astype(np.uint8)
                rescaled_mask_image = rescale_mask_V2(mask[0], curr_trans_params[0], original_shape=(256, 256))
                rescaled_mask_image = get_contour(np.array(rescaled_mask_image)[..., 0].astype(np.uint8))
                rescaled_mask_image = cv2.GaussianBlur(rescaled_mask_image, (21, 21), 21)[..., None]
                # rescaled_mask_image = transforms.functional.gaussian_blur(rescaled_mask_image, kernel_size=(11, 11))

                rendered_face_mask_img_tensor = torch.FloatTensor(np.array(rescaled_mask_image)) / 255.0
                rendered_face_mask_img_tensor = rendered_face_mask_img_tensor.permute(2, 0, 1) # (1, H, W)

                ## Get the binary mask image from the 3DMM rendered face image
                # rendered_face_mask_img = get_masked_region(np.array(rescaled_rendered_image))[..., None]
                # rendered_face_mask_img_tensor = torch.FloatTensor(rendered_face_mask_img) / 255.0
                # rendered_face_mask_img_tensor = rendered_face_mask_img_tensor.permute(2, 0, 1) # (1, H, W)
                
                ## Get the blended face image
                blended_img_tensor = source_image * (1 - rendered_face_mask_img_tensor) + \
                                     rendered_image * rendered_face_mask_img_tensor
                data['blended_image'].append(blended_img_tensor)
                data['masked_image'].append(rendered_face_mask_img_tensor.repeat(3, 1, 1))
                data['rendered_image'].append(rendered_image)
            
        data['video_name'] = self.obtain_name(video_item['video_name'], source_video_item['video_name'])
        return data
    
    def random_video(self, target_video_item):
        target_person_id = target_video_item['person_id']
        assert len(self.person_ids) > 1 
        source_person_id = np.random.choice(self.person_ids)
        if source_person_id == target_person_id:
            source_person_id = np.random.choice(self.person_ids)
        source_video_index = np.random.choice(self.idx_by_person_id[source_person_id])
        source_video_item = self.video_items[source_video_index]
        return source_video_item

    def find_crop_norm_ratio(self, source_coeff, target_coeffs):
        alpha = 0.3
        exp_diff = np.mean(np.abs(target_coeffs[:,80:144] - source_coeff[:,80:144]), 1)
        angle_diff = np.mean(np.abs(target_coeffs[:,224:227] - source_coeff[:,224:227]), 1)
        index = np.argmin(alpha*exp_diff + (1-alpha)*angle_diff)
        crop_norm_ratio = source_coeff[:,-3] / target_coeffs[index:index+1, -3]
        return crop_norm_ratio
   
    def transform_semantic(self, semantic, frame_index, crop_norm_ratio):
        index = self.obtain_seq_index(frame_index, semantic.shape[0])
        coeff_3dmm = semantic[index,...]
        id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:,80:144] #expression
        tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:,224:227] #euler angles for pose
        gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:,254:257] #translation
        crop = coeff_3dmm[:,257:300] #crop param

        if self.cross_id and self.norm_crop_param:
            crop[:, -3] = crop[:, -3] * crop_norm_ratio

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        coeff_3dmm_all = np.concatenate([id_coeff, ex_coeff, tex_coeff, angles, gamma, translation, crop], 1)
        return torch.Tensor(coeff_3dmm).permute(1,0), torch.Tensor(coeff_3dmm_all).permute(1,0)

    def obtain_name(self, target_name, source_name):
        if not self.cross_id:
            return target_name
        else:
            source_name = os.path.splitext(os.path.basename(source_name))[0]
            return source_name+'_to_'+target_name
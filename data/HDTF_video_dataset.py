'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-09 09:32:07
Email: haimingzhang@link.cuhk.edu.cn
Description: The dataset class to infer in HDTF dataset.
'''

import os
import os.path as osp
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from collections import defaultdict

from data.vox_dataset import format_for_lmdb
from data.HDTF_dataset import HDTFDataset
from .face_utils import get_masked_region, rescale_mask_V2, get_contour


class HDTFVideoDataset(HDTFDataset):
    def __init__(self, opt, is_inference):
        super(HDTFVideoDataset, self).__init__(opt, is_inference)
        
        self.video_index = -1
        
        self.cross_id = False
        # whether normalize the crop parameters when performing cross_id reenactments
        # set it as "True" always brings better performance
        self.norm_crop_param = True

        self.use_cross_expression = True

        self.all_videos_dir = list(self.video_name_to_imgs_list_dict.keys())

        self.fetch_length = 200

    def __len__(self):
        return len(self.video_name_to_imgs_list_dict)

    def load_next_video(self):
        self.video_index += 1
        
        video_name = self.all_videos_dir[self.video_index]
        num_frames = self.video_frame_length_dict[video_name]

        start_index = random.choice(range(num_frames - self.fetch_length))

        data = defaultdict(list)
        for frame_index in range(start_index, start_index + self.fetch_length):
            ## Read the source image and source semantics
            source_img_path = osp.join(self.data_root, video_name, "face_image", f"{frame_index:06d}.jpg")
            source_image = Image.open(source_img_path).convert("RGB")
            source_image = self.transform(source_image)
            data['source_image'].append(source_image)

            ref_img_idx = 0
            ref_img_path = osp.join(self.data_root, video_name, "face_image", f"{ref_img_idx:06d}.jpg")
            reference_image = Image.open(ref_img_path).convert("RGB")
            data['reference_image'].append(self.transform(reference_image))

            ## Get the 3DMM rendered face image
            index_seq_window = self.get_index_seq_window(frame_index, num_frames)

            coeff_3dmm_list, semantic_params_list = [], []
            for i in index_seq_window:
                mat_fp = osp.join(self.data_root, video_name, "deep3dface", f"{i:06d}.mat")
                semantic_params, coeff_3dmm_all = self.read_face3dmm_params(mat_fp)
                coeff_3dmm_list.append(coeff_3dmm_all)
                semantic_params_list.append(semantic_params)

            coeff_3dmm_all_tensor = torch.Tensor(np.concatenate(coeff_3dmm_list, axis=0)).permute(1,0)
            source_semantics = torch.Tensor(np.concatenate(semantic_params_list, axis=0)).permute(1,0)
            data['source_semantics'].append(source_semantics)

                ## Get the rendered face image
            curr_semantics = coeff_3dmm_all_tensor[:, 13:14].permute(1, 0) # (1, 260)
            curr_face3dmm_params = curr_semantics[:, :257] # (1, 257)
            curr_trans_params = curr_semantics[:, -3:]

            rendered_image_numpy, mask = self.face_renderer.compute_rendered_face(curr_face3dmm_params, None)
            # get the rescaled_rendered_image (256, 256, 3)
            rescaled_rendered_image = rescale_mask_V2(rendered_image_numpy[0], curr_trans_params[0], original_shape=(512, 512))
            rescaled_rendered_image = self.transform(rescaled_rendered_image)
            data['rendered_image'].append(rescaled_rendered_image)

            ## get the rescaled mask image
            mask = mask.permute(0, 2, 3, 1).cpu().numpy() * 255 # (B, 224, 224, 1)
            mask = mask.astype(np.uint8)
            rescaled_mask_image = rescale_mask_V2(mask[0], curr_trans_params[0], original_shape=(512, 512))
            rescaled_mask_image = get_contour(np.array(rescaled_mask_image)[..., 0].astype(np.uint8))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            rescaled_mask_image = cv2.erode(rescaled_mask_image, kernel=kernel, iterations=1)

            rendered_face_mask_img_tensor = torch.FloatTensor(np.array(rescaled_mask_image)) / 255.0
            rendered_face_mask_img_tensor = rendered_face_mask_img_tensor[..., None].permute(2, 0, 1) # (1, H, W)
            
            ## Get the blended face image
            blended_img_tensor = source_image * (1 - rendered_face_mask_img_tensor) + \
                                 rescaled_rendered_image * rendered_face_mask_img_tensor
            
            data['blended_image'].append(blended_img_tensor)
            data['masked_image'].append(rendered_face_mask_img_tensor.repeat(3, 1, 1))
        
        data_output = {}
        data_output['video_name'] = self.obtain_name(video_name, video_name)
        data_output.update(data)
        return data_output
    
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
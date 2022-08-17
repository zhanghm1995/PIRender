'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-09 09:32:07
Email: haimingzhang@link.cuhk.edu.cn
Description: Load the demo dataset for TalkingFaceFormer.
'''

import os
import os.path as osp
from glob import glob
import collections
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import torch
from torch import Tensor
from torch.nn import functional as F
import torchvision.transforms as transforms
from collections import defaultdict
from torch.utils.data import Dataset
from scipy.io import loadmat

from visualizer.render_utils import MyMeshRender

from .face_utils import get_masked_region, rescale_mask_V2, get_contour


def get_coeff_vector(face_params_dict, key_list=None, reset_list=None):
    """Get coefficient vector from Deep3DFace_Pytorch results

    Args:
        face_params_dict (dict): the dictionary contains reconstructed 3D face

    Returns:
        [np.ndarray]: 1x257
    """
    if key_list is None:
        keys_list = ['id', 'exp', 'tex', 'angle', 'gamma', 'trans']
    else:
        keys_list = key_list

    coeff_list = []
    for key in keys_list:
        if reset_list is not None and key in reset_list:
            value = np.zeros_like(face_params_dict[key])
            coeff_list.append(value)
        else:
            coeff_list.append(face_params_dict[key])
    
    coeff_res = np.concatenate(coeff_list, axis=1)
    return coeff_res


def read_face3dmm_params(file_path, need_crop_params=False):
    """Read the 3dmm face parameters from mat file

    Args:
        file_path (_type_): _description_
        need_crop_params (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    assert file_path.endswith(".mat")

    file_mat = loadmat(file_path)
    coeff_3dmm = get_coeff_vector(file_mat)
    
    if need_crop_params:
        crop_param = file_mat['transform_params']
        _, _, ratio, t0, t1 = np.hsplit(crop_param.astype(np.float32), 5)
        crop_param = np.concatenate([ratio, t0, t1], 1)
        coeff_3dmm = np.concatenate([coeff_3dmm, crop_param], axis=1)

    return coeff_3dmm


class HDTFDemoDataset(Dataset):
    def __init__(
        self, 
        data_root,
        video_name,
        pred_dir):

        super(HDTFDemoDataset, self).__init__()

        ## Get all key frames
        self.data_root = data_root
        self.video_dir = osp.join(self.data_root, video_name)

        self.image_paths = sorted(glob(osp.join(self.video_dir, "face_image", "*.jpg")))

        self.pred_paths = sorted(glob(osp.join(pred_dir, "*.mat")))

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])
        
        self.face_renderer = MyMeshRender()

        self.half_mask = np.zeros((224, 224, 1), dtype=np.uint8)
        self.half_mask[:128, ...] = 255

        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 15))

    def __len__(self):
        return len(self.pred_paths)

    def __getitem__(self, index):
        pred_3d_face_mat_path = self.pred_paths[index]

        data = {}
        ## Read the source image and source semantics
        # source_image_path = osp.join(self.video_dir, "face_image", file_name + ".jpg")
        source_image_path = self.image_paths[index]
        source_image = Image.open(source_image_path).convert("RGB")
        source_image = self.transform(source_image)
        data['source_image'] = source_image

        ref_img_idx = 0
        ref_img_path = self.image_paths[ref_img_idx]
        reference_image = Image.open(ref_img_path).convert("RGB")
        data['reference_image'] = self.transform(reference_image)

        ## Read the predicted 3D face coefficients
        coeff_3dmm = read_face3dmm_params(pred_3d_face_mat_path, need_crop_params=True)
        coeff_3dmm = torch.from_numpy(coeff_3dmm) # (1, 260)
        curr_face3dmm_params = coeff_3dmm[:, :257] # (1, 257)
        curr_trans_params = coeff_3dmm[:, -3:]

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

        data['masked_image'] = rendered_face_mask_img_tensor.repeat(3, 1, 1)
        
        return data


    def __getitem_image__(self, index):
        pred_3d_face_image_path = self.pred_paths[index]
        file_name = osp.basename(pred_3d_face_image_path)[:-4]

        data = {}
        ## Read the source image and source semantics
        # source_image_path = osp.join(self.video_dir, "face_image", file_name + ".jpg")
        source_image_path = self.image_paths[index]
        source_image = Image.open(source_image_path).convert("RGB")
        source_image = self.transform(source_image)
        data['source_image'] = source_image

        ref_img_idx = 0
        ref_img_path = self.image_paths[ref_img_idx]
        reference_image = Image.open(ref_img_path).convert("RGB")
        data['reference_image'] = self.transform(reference_image)

        ## Read the rendered image
        rendered_image_path = pred_3d_face_image_path
        rendered_image = cv2.imread(rendered_image_path)

        rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)
        rescaled_rendered_image = self.transform(rendered_image)
        
        rescaled_mask_image = get_masked_region(rendered_image)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        # rescaled_mask_image = cv2.erode(rescaled_mask_image, kernel=kernel, iterations=1)
        # rescaled_mask_image = cv2.dilate(rescaled_mask_image, kernel=kernel, iterations=1)


        rendered_face_mask_img_tensor = torch.FloatTensor(np.array(rescaled_mask_image)) / 255.0
        rendered_face_mask_img_tensor = rendered_face_mask_img_tensor[..., None].permute(2, 0, 1) # (1, H, W)

        blended_img_tensor = source_image * (1 - rendered_face_mask_img_tensor) + \
                             rescaled_rendered_image * rendered_face_mask_img_tensor

        data['rendered_image'] = rescaled_rendered_image
        data['masked_image'] = rendered_face_mask_img_tensor.repeat(3, 1, 1)
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
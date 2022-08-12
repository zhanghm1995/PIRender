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
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import cv2
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from collections import defaultdict
from torch.utils.data import Dataset

from .face_utils import get_masked_region, rescale_mask_V2, get_contour


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

        self.pred_paths = sorted(glob(osp.join(pred_dir, "*.png")))

        # assert len(self.image_paths) == len(self.pred_paths)
        
        self.fetch_length = 200

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])

    def __len__(self):
        return len(self.pred_paths)

    def __getitem__(self, index):
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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        rescaled_mask_image = cv2.erode(rescaled_mask_image, kernel=kernel, iterations=1)

        rendered_face_mask_img_tensor = torch.FloatTensor(np.array(rescaled_mask_image)) / 255.0
        rendered_face_mask_img_tensor = rendered_face_mask_img_tensor[..., None].permute(2, 0, 1) # (1, H, W)

        blended_img_tensor = source_image * (1 - rendered_face_mask_img_tensor) + \
                             rescaled_rendered_image * rendered_face_mask_img_tensor

        data['rendered_image'] = rescaled_rendered_image
        data['masked_image'] = rendered_face_mask_img_tensor.repeat(3, 1, 1)
        data['blended_image'] = blended_img_tensor
        
        return data
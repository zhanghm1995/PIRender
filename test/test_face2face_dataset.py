'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-06 21:12:04
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import sys
sys.path.append('./')
sys.path.append('../')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import torchvision
import numpy as np
from easydict import EasyDict

import data as Dataset
from config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/face_to_face.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    return args


args = parse_args()
opt = Config(args.config, args, is_train=True)
opt.data.train.distributed = False
print(opt)


def test_VoxFace2FaceDataset():
    # create a dataset
    val_dataset, train_dataset = Dataset.get_train_val_dataloader(opt.data)
    print(len(train_dataset))


    for batch in train_dataset:
        print(type(batch))
        for key, value in batch.items():
            print(key, value.shape)
        
        # face_3dmm_params = batch['source_semantics']
        batch['source_image'] = (batch['source_image'] + 1.0) / 2.0
        batch['rendered_image'] = (batch['rendered_image'] + 1.0) / 2.0
        batch['blended_image'] = (batch['blended_image'] + 1.0) / 2.0

        torchvision.utils.save_image(batch['source_image'], 'source_image.png')
        torchvision.utils.save_image(batch['rendered_image'], 'rendered_image.png')
        torchvision.utils.save_image(batch['blended_image'], 'blended_image.png')

        break


def test_VoxFace2FaceVideoDataset():
    from data.vox_face_to_face_video_dataset import VoxFace2FaceVideoDataset
    
    opt = EasyDict()
    opt.type = "data.vox_dataset::VoxDataset"
    opt.path = "./dataset/vox_lmdb_demo"
    opt.resolution = 256
    opt.semantic_radius = 13
    opt.train = EasyDict()
    opt.val = EasyDict()
    opt.train.batch_size = 5
    opt.train.distributed = False
    opt.val.batch_size = 5
    opt.val.distributed = False

    dataset = VoxFace2FaceVideoDataset(opt, is_inference=True)
    print(len(dataset))

    for video_index in range(dataset.__len__()):
        data = dataset.load_next_video()
        print(data['video_name'])
        break


if __name__ == "__main__":
    test_VoxFace2FaceVideoDataset()
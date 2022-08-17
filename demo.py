'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-12 19:30:44
Email: haimingzhang@link.cuhk.edu.cn
Description: The demo script for TalkingFaceFormer
'''

import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import time
import numpy as np

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import argparse

from util.logging import init_logging, make_logging_dir
from util.distributed import init_dist
from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer
from util.distributed import master_only_print as print
from data.vox_face_to_face_video_dataset import VoxFace2FaceVideoDataset
from data.HDTF_demo_dataset import HDTFDemoDataset

from config import Config
from easydict import EasyDict


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/face_to_face.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--checkpoints_dir', default='result',
                        help='Dir for saving logs and models.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--cross_id', action='store_true')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()
    return args


def write2video(results_dir, *video_list):
    cat_video=None

    for video in video_list:
        video_numpy = video[:,:3,:,:].cpu().float().detach().numpy()
        video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        video_numpy = video_numpy.astype(np.uint8)
        cat_video = np.concatenate([cat_video, video_numpy], 2) if cat_video is not None else video_numpy

    image_array=[]
    for i in range(cat_video.shape[0]):
        image_array.append(cat_video[i]) 

    out_name = results_dir+'.mp4' 
    _, height, width, layers = cat_video.shape
    size = (width,height)
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    for i in range(len(image_array)):
        out.write(image_array[i][:,:,::-1])
    out.release() 


def save_images(save_dir, image_batch, start_index):
    image_batch = image_batch[:, :3, :, :].cpu().float().detach().numpy()
    image_numpy = (np.transpose(image_batch, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)

    for i in range(image_numpy.shape[0]):
        image = image_numpy[i][..., ::-1]
        cv2.imwrite(osp.join(save_dir, '{:06d}.jpg'.format(start_index + i)), image)


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=False)

    if not args.single_gpu:
        opt.local_rank = args.local_rank
        init_dist(opt.local_rank)    
        opt.device = torch.cuda.current_device()
    
    opt.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create a visualizer
    date_uid, logdir = init_logging(opt)
    opt.logdir = logdir
    make_logging_dir(logdir, date_uid)

    # create a model
    net_G, net_G_ema, opt_G, sch_G \
        = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_G_ema, \
                          opt_G, sch_G, None)

    current_epoch, current_iteration = trainer.load_checkpoint(
        opt, args.which_iter)                          
    net_G = trainer.net_G_ema.eval()

    save_dir = os.path.join(
        args.output_dir, 
        )
    os.makedirs(save_dir, exist_ok=True)

    ## Build dataset
    # config = EasyDict()
    # config.data_root = "./dataset/HDTF_face3dmmformer/val"
    # config.video_name = "WDA_JeanneShaheen0_000"
    # config.pred_dir = "/home/zhanghm/Research/Face/PIRender/WDA_JeanneShaheen0_000_condition_WDA_ChrisMurphy0_000"
    # config.pred_dir = "./dataset/HDTF_face3dmmformer/train/WDA_KimSchrier_000/deep3dface"

    dataset = HDTFDemoDataset(**opt.hdtf_data_demo)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0)
    print(f"The dataset length is {len(dataset)}, The dataloader length is {len(dataloader)}")
    
    ## Start inference
    with torch.no_grad():
        count = 6600
        for data in tqdm(dataloader):
            for key, value in data.items():
                data[key] = value.cuda()

            blended_image, reference_image = data['blended_image'], data['reference_image']
            source_image = data['source_image']
            
            output_dict = net_G(reference_image, blended_image)
            output_images = output_dict['fake_image'].cpu().clamp_(-1, 1)
            # output_images = data['masked_image'].cpu()
            # output_images = data['blended_image'].cpu()

            ## Save images
            save_images(save_dir, output_images, count)

            batch_size = blended_image.shape[0]
            count += batch_size

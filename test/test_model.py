'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-08 14:14:06
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''


import sys
sys.path.append("./")
sys.path.append("../")

import argparse
import torch

from config import Config
from generators.face_to_face_model import Face2FaceGenerator


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
opt = Config(args.config, args=args, is_train=True)

net_G = Face2FaceGenerator(**opt.gen.param)

image_size = 256
input_image = torch.randn(1, 3, image_size, image_size)
rendered_image = torch.randn(1, 3, image_size, image_size)
driving_source = torch.randn(1, 73, 27)

output = net_G(input_image, rendered_image, driving_source)
print(output['fake_image'].shape)

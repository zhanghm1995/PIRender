'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-06 15:09:31
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

import data as Dataset
from config import Config
from visualizer.render_utils import MyMeshRender


def rescale_mask_V2(input_mask: np.array, transform_params: list, original_shape: tuple):
    """
    Uncrops and rescales (i.e., resizes) the given scaled and cropped mask back to the
    resolution of the original image using the given transformation parameters.
    """
    original_image_width, original_image_height = original_shape
    s = np.float64(transform_params[0])
    t = transform_params[1:]
    target_size = 224.0

    scaled_image_w = (original_image_width * s).astype(np.int32)
    scaled_image_h = (original_image_height * s).astype(np.int32)
    left = (scaled_image_w/2 - target_size/2 + float((t[0] - original_image_width/2)*s)).astype(np.int32)
    up = (scaled_image_h/2 - target_size/2 + float((original_image_height/2 - t[1])*s)).astype(np.int32)

    # Parse transform params.
    mask_scaled = Image.new('RGB', (scaled_image_w, scaled_image_h), (0, 0, 0))
    mask_scaled.paste(Image.fromarray(input_mask), (left, up))
    
    # Rescale the uncropped mask back to the resolution of the original image.
    uncropped_and_rescaled_mask = mask_scaled.resize((original_image_width, original_image_height), 
                                                      resample=Image.Resampling.BICUBIC)
    return uncropped_and_rescaled_mask


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default='./config/face.yaml')
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

# create a dataset
val_dataset, train_dataset = Dataset.get_train_val_dataloader(opt.data)
print(len(train_dataset))

face_3dmm_params = None
for batch in train_dataset:
    print(type(batch))
    for key, value in batch.items():
        print(key, value.shape)
    
    face_3dmm_params = batch['source_semantics']
    break

face_renderer = MyMeshRender()

## Visualize a single image
# curr_face3dmm_params = face_3dmm_params[0:1, :, 13]
# print(curr_face3dmm_params.shape, curr_face3dmm_params[:, -3:])

# dummy_face3dmm_params = torch.zeros(1, 257)
# dummy_face3dmm_params[:, 80:144] = curr_face3dmm_params[:, :64]
# dummy_face3dmm_params[:, 224:227] = curr_face3dmm_params[:, 64:67]
# dummy_face3dmm_params[:, 254:257] = curr_face3dmm_params[:, 67:70]


# image = face_renderer.compute_rendered_face(dummy_face3dmm_params, None)
# print(image.shape, image.max())

# Image.fromarray(image[0]).save('test.png')


## save image and rendered image together
for batch in train_dataset:
    source_image = batch['source_image']
    curr_face3dmm_params = batch['source_semantics'][:, :, 13]
    source_image = (source_image + 1.0) / 2.0

    # source_image = nn.functional.interpolate(source_image, size=(224, 224), mode='bilinear', align_corners=False)

    batch_size = source_image.shape[0]

    dummy_face3dmm_params = torch.zeros(batch_size, 257)
    # dummy_face3dmm_params[:, 80:144] = curr_face3dmm_params[:, :64]
    # dummy_face3dmm_params[:, 224:227] = curr_face3dmm_params[:, 64:67]
    # dummy_face3dmm_params[:, 254:257] = curr_face3dmm_params[:, 67:70]
    dummy_face3dmm_params[:, :] = curr_face3dmm_params[:, :257]
    image = face_renderer.compute_rendered_face(dummy_face3dmm_params, None)

    transform_params = curr_face3dmm_params[:, -3:]

    if transform_params is not None:
            ## TODO: cannot apply in batch processing currently
            trans_img = []
            for i in range(image.shape[0]):
                img = image[i]
                trans_vec = transform_params[i]

                tmp_img = rescale_mask_V2(img, trans_vec, original_shape=(256, 256))
                trans_img.append(tmp_img)
            
            vis_image = np.stack(trans_img)

    rendered_face = face_renderer.pred_face
    print(rendered_face.shape, vis_image.shape)

    vis_image = torch.from_numpy(vis_image / 255.0)
    vis_image = vis_image.permute(0, 3, 1, 2)
    print(vis_image.shape)

    blended_image = source_image * 0.6 + vis_image * 0.4

    vis_images = torch.cat((source_image, vis_image, blended_image), dim=0)
    torchvision.utils.save_image(vis_images, 'test7.png', nrow=batch_size, padding=0)
    break

'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-05 23:10:20
Email: haimingzhang@link.cuhk.edu.cn
Description: The utility functions for face processing
'''

import numpy as np
import cv2
from PIL import Image


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


def get_contour(im):
    contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(im)
    out = cv2.fillPoly(out, contours, 255)

    return out


def get_masked_region(mask_img):
    gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    thresh = cv2.dilate(thresh, kernel=kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel=kernel, iterations=1)

    contour = get_contour(thresh)
    # contour = cv2.GaussianBlur(contour,(9, 9), 0)
    contour = cv2.erode(contour, np.ones((11, 11), np.uint8), iterations=1)
    return contour


def rescale_mask_V2(input_mask: np.array, transform_params: list, original_shape: tuple):
    """
    Uncrops and rescales (i.e., resizes) the given scaled and cropped mask back to the
    resolution of the original image using the given transformation parameters.
    """
    if input_mask.shape[2] != 3:
        input_mask = np.tile(input_mask, (1, 1, 3))
    
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
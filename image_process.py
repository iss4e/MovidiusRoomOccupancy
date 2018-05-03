#!/usr/bin/python3

import numpy
import cv2
from scipy.misc import imrotate

def crop_image(img):
    """ Crops the raw raspberry pi image to the office lab area """
    img = imrotate(img, -4.5) 
    img = img[ 592 : 1243 + 592, 736: 2354 + 736]
    return img

def split_to_five(img):
    """ Splits the cropped office image into four desk areas and an area directly under the camera """

    height, width, channels = img.shape

    top_left = img[0:height//2, 0:width//2]
    top_right = img[0:height//2, width//2 : width]

    bottom_left = img[height//2: height-1, 0:width//2]
    bottom_right = img[height//2: height-1, width//2: width]

    center_factor = 104

    center = img[height//4:3*height//4-1, width//4 - center_factor : 3*width//4 - center_factor] 

    split_images = [top_left, top_right, bottom_left, bottom_right, center]

    split_images = list(map(transform_img, split_images))
    return split_images

def transform_img(img, img_width=227, img_height=227):
    """ Performs histogram equalization and adds padding to the image when resizing """

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    old_size = img.shape[:2]
    desired_size = img_width
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    img = cv2.resize(img, (new_size[1], new_size[0]), interpolation = cv2.INTER_CUBIC)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h - (delta_h//2)
    left, right = delta_w//2, delta_w - (delta_w//2)
    color = [0,0,0]

    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img

def normalize_img(input_img,mean_img, img_width=227,img_height=227):
    """ Subtracts the mean image from the input image"""

    input_img=cv2.resize(input_img,(img_width,img_height))
    input_img = input_img.astype(numpy.float32)
    input_img[:,:,0] = (input_img[:,:,0] - mean_img[0])
    input_img[:,:,1] = (input_img[:,:,1] - mean_img[1])
    input_img[:,:,2] = (input_img[:,:,2] - mean_img[2])
    return input_img


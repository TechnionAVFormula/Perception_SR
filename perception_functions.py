import os
import ast
import cv2
import numpy as np
from PIL import Image
# from detect import get_BB_from_img
from config import CONFIG
from config import ConfigEnum
from PIL import Image, ImageDraw, ImageFont
from utils.utils import *
from utils.nms import nms

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision

import logging

if (CONFIG  == ConfigEnum.REAL_TIME) or (CONFIG == ConfigEnum.COGNATA_SIMULATION):
    from pyFormulaClient import messages
elif ( CONFIG == ConfigEnum.LOCAL_TEST):
    from pyFormulaClientNoNvidia import messages
else:
    raise NameError('User Should Choose Configuration from config.py')

from timeit import default_timer as timer


# def get_cones_from_camera(width, height, pixels, weights_path, model_cfg):
#     # convert from bit representation to RGB + depth format
#     img_RGB = convert_img_bits_to_RGBD(width, height, pixels)

#     # Detect cones in input image via YOLO
#     BB_list = get_BB_from_img(img_RGB,weights_path,model_cfg,conf_thres = 0.8,nms_thres = 0.25,xy_loss = 2,wh_loss = 1.6,no_object_loss = 25,object_loss=0.1,vanilla_anchor = False)

#     # classify detected cone to types
#     for BB in BB_list:
#         cone_color = predict_cone_color(img_RGB,BB)
#         # cone_depth = predict_cone_depth(img_depth,BB)
#         # BB = [x,y,h,w,type,depth]
#         BB.append(cone_color)
#     # BB_list = [BB_1,BB_2,...,BB_N]

#     img_RGB.close()

#     return BB_list

def cones_detection(width, height, target_img, model, device, conf_thres, nms_thres):

    img_conversion_time = timer()
    # convert from bit representation to RGB + depth format
    img_RGB = convert_img_bits_to_RGBD(width, height, target_img)
    w = width
    h = height
    img_conversion_time = timer() - img_conversion_time

    preprocessing_time = timer()
    new_width, new_height = model.img_size()
    pad_h, pad_w, ratio = calculate_padding(h, w, new_height, new_width)
    img = torchvision.transforms.functional.pad(img_RGB, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
    img = torchvision.transforms.functional.resize(img, (new_height, new_width))
    img = torchvision.transforms.functional.to_tensor(img)
    img = img.unsqueeze(0)
    preprocessing_time = timer() - preprocessing_time
    
    with torch.no_grad():
        img_to_device_time = timer()
        model.eval()
        img = img.to(device, non_blocking=True)
        img_to_device_time = timer() - img_to_device_time
        # output,first_layer,second_layer,third_layer = model(img)

        model_device_time = timer()
        output = model(img)
        model_device_time = timer() - model_device_time

        post_processing_time = timer()
        # Tresholding detections
        for detections in output:
            detections = detections[detections[:, 4] > conf_thres]
            box_corner = torch.zeros((detections.shape[0], 4), device=detections.device)
            xy = detections[:, 0:2]
            wh = detections[:, 2:4] / 2
            box_corner[:, 0:2] = xy - wh
            box_corner[:, 2:4] = xy + wh
            probabilities = detections[:, 4]
            nms_indices = nms(box_corner, probabilities, nms_thres)
            main_box_corner = box_corner[nms_indices]
            probabilities_nms = probabilities[nms_indices]
            if nms_indices.shape[0] == 0:  
                continue
        post_processing_time = timer() - post_processing_time
        
        BB_list_creation_time = timer()
        # BB_list = [BB_1,BB_2,...,BB_N]
        BB_list = []
        # Extracting bounding boxes 
        for i in range(len(main_box_corner)):
            x0 = main_box_corner[i, 0].to('cpu').item() / ratio - pad_w
            y0 = main_box_corner[i, 1].to('cpu').item() / ratio - pad_h
            x1 = main_box_corner[i, 2].to('cpu').item() / ratio - pad_w
            y1 = main_box_corner[i, 3].to('cpu').item() / ratio - pad_h 
            Pr = probabilities_nms[i]
            BB = { 'u': round(x0), 'v' : round(y0), 'h' : round(y1 - y0), 'w' : round(x1 - x0), 'Pr' : Pr}  # x, y, h, w, Pr
            # u, v - left top bounding box position in image plain
            # w, h - width and height of bounding box in pixels
            BB_list.append(BB)
        BB_list_creation_time = timer() - BB_list_creation_time
        
        color_prediction_time = timer()
        # classify detected cone to types
        for BB in BB_list:
            cone_color = predict_cone_color(img_RGB,BB)
            # cone_depth = predict_cone_depth(img_depth,BB)
            # BB = ['u','v','h','w','pr,'type'] ==>  (,'depth']) in the future we will have a deapth image
            BB['type'] = cone_color
            # BB.append(cone_color)
        color_prediction_time = timer() - color_prediction_time
        

        # img_RGB.close()
        logging.debug("Image conversion took: %d [ms]", 1000 *img_conversion_time)
        logging.debug("Preprocessing took: %d [ms]", 1000 *preprocessing_time)
        logging.debug("Image to device took: %d [ms]", 1000 *img_to_device_time)
        logging.debug("Running model took: %d [ms]", 1000 *model_device_time)
        logging.debug("Postprocessing took: %d [ms]", 1000 *post_processing_time)
        logging.debug("Creating bounding box list took: %d [ms]", 1000 *BB_list_creation_time)
        logging.debug("Predicting color took: %d [ms]", 1000 *color_prediction_time)

        return BB_list



def convert_img_bits_to_RGBD(width, height, pixels):
    # convert bit format to RGB
    img_RGB = Image.frombytes("RGB", (width, height), pixels, 'raw', 'RGBX', 0,-1)
    return img_RGB

def cut_cones_from_img(target_img, BB):
    x = BB['u']
    y = BB['v']
    w = BB['w']
    h = BB['h']
    crop_img = target_img.crop((x, y,x+w,y+h))
    return crop_img

def predict_cone_color(target_img, BB):

    frame = cut_cones_from_img(target_img, BB)
    frame = np.array(frame) # convert from PIL to cv
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # General mask
    low_general = np.array([0, 50, 80])
    high_general = np.array([179, 255, 255])
    general_mask = cv2.inRange(hsv_frame, low_general, high_general)
    general = cv2.bitwise_and(frame, frame, mask=general_mask)
    # cv2.imshow("General", general)

    # Yellow color
    low_yellow = np.array([94, 80, 2])
    high_yellow = np.array([126, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(general, general, mask=yellow_mask)
    # cv2.imshow("Yellow", yellow)

    # Blue color
    low_blue = np.array([5, 50, 50])
    high_blue = np.array([10, 295, 295])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(general, general, mask=blue_mask)
    # cv2.imshow("Blue", blue)

    # Orange color
    low_orange = np.array([20, 190, 20])
    high_orange = np.array([30, 255, 255])
    orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
    orange = cv2.bitwise_and(general, general, mask=orange_mask)
    # cv2.imshow("Orange", orange)

    final_frame = cv2.hconcat((frame, yellow, blue, orange))
    # cv2.imshow("final_frame", final_frame)

    n_white_pix_yellow = np.sum(yellow_mask == 255)
    n_white_pix_blue = np.sum(blue_mask == 255)
    n_white_pix_orange = np.sum(orange_mask == 255)
    n_white_pix = [n_white_pix_yellow, n_white_pix_blue, n_white_pix_orange]
    # print('Number of white pixels: Yellow_mask = ', n_white_pix_yellow, " | Blue_mask = ", n_white_pix_blue, " | Orange_mask = ", n_white_pix_orange)
    max_value = max(n_white_pix)
    max_idx = n_white_pix.index(max_value)

    if max_idx == 0: # cone is yellow
        return messages.perception.Yellow
    elif max_idx == 1: # cone is blue
        return messages.perception.Blue
    else: # cone is orange
        return messages.perception.Orange


# def predict_cone_color(target_img, BB):
#     frame = cut_cones_from_img(target_img, BB)
#     frame = np.array(frame) # convert from PIL to cv
#     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     imgB = cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2HSV)
#     imgY = cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2HSV)

#     blueLow = np.array([90, 50, 50])
#     blueHigh = np.array([150, 255, 255])
#     yellowLow = np.array([20, 100, 100])
#     yellowHigh = np.array([40, 255, 255])

#     maskBlue = cv2.inRange(imgB, blueLow, blueHigh)
#     maskBlue2 = cv2.dilate(maskBlue, np.ones((5, 5), np.uint8))
#     maskBlue3 = cv2.erode(maskBlue2, np.ones((5, 5), np.uint8))
#     maskYellow = cv2.inRange(imgY, yellowLow, yellowHigh)
#     maskYellow2 = cv2.dilate(maskYellow, np.ones((5, 5), np.uint8))
#     maskYellow3 = cv2.erode(maskYellow2, np.ones((5, 5), np.uint8))
#     outputBlue = cv2.bitwise_and(imgB, imgB, mask=maskBlue3) 
#     outputYellow = cv2.bitwise_and(imgY, imgY, mask=maskYellow3) 

#     tempBlue = cv2.cvtColor(outputBlue, cv2.COLOR_BGR2GRAY)
#     tempYellow = cv2.cvtColor(outputYellow, cv2.COLOR_BGR2GRAY)
#     edgedBlue = cv2.Canny(tempBlue, 30, 300)
#     edgedYellow = cv2.Canny(tempYellow, 30, 300)
    

#     # final_frame = cv2.hconcat((frame, outputYellow, outputBlue))
#     # cv2.imshow("final_frame", final_frame)

#     n_white_pix_yellow = np.sum(yellow_mask == 255)
#     n_white_pix_blue = np.sum(blue_mask == 255)
#     n_white_pix = [n_white_pix_yellow, n_white_pix_blue]
#     # print('Number of white pixels: Yellow_mask = ', n_white_pix_yellow, " | Blue_mask = ", n_white_pix_blue, " | Orange_mask = ", n_white_pix_orange)
#     max_value = max(n_white_pix)
#     max_idx = n_white_pix.index(max_value)

#     if max_idx == 0: # cone is yellow
#         return messages.perception.Yellow
#     elif max_idx == 1: # cone is blue
#         return messages.perception.Blue



def predict_cone_depth(img_depth, BB):
    frame = img_depth
    array = np.array(frame)
    max_pixel_value = array.max()
    return max_pixel_value

def get_BB_img_point(img_cone):
    # return representative point in BB (center of BB at the moment)
    x, y, h, w, pr, color = img_cone.values()
    return int(x+w/2), int(y+h/2), color

def draw_results_on_image(img, BB_list, type_map):

    img_with_boxes = img
    draw = ImageDraw.Draw(img_with_boxes)
    font = ImageFont.load_default()

    for i in range(len(BB_list)):
        # extract BB features:
        x0 = BB_list[i][0]
        y0 = BB_list[i][1]
        h = BB_list[i][2]
        w = BB_list[i][3]
        raw_type = BB_list[i][4]
        # build parameters
        color = type_map[raw_type-1]
        text = f"({i}) {color}"
        x1 = x0 + w
        y1 = y0 + h
        # draw BB + indicative text
        draw.rectangle((x0, y0, x1, y1), outline=color)
        w_text, h_text = font.getsize(text)
        draw.rectangle((x0, y0-h_text, x0 + w_text, y0), fill=color)
        draw.text((x0, y0-h_text),text , fill=(0, 0, 0, 128))

    return img_with_boxes

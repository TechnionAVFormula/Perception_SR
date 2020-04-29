#!/usr/bin/python3

import argparse
import os
from os.path import isfile, join
import random
import tempfile
import time
import copy
import multiprocessing
import subprocess
import shutil
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from PIL import Image, ImageDraw

import torchvision
from models import Darknet
from utils.datasets import ImageLabelDataset
from utils.nms import nms
from utils.utils import xywh2xyxy, calculate_padding

import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

detection_tmp_path = "/tmp/detect/"


def get_BB_from_img(target_img,weights_path,model_cfg,conf_thres,nms_thres,xy_loss,wh_loss,no_object_loss,object_loss,vanilla_anchor):

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    random.seed(0)
    torch.manual_seed(0)
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    model = Darknet(config_path=model_cfg,xy_loss=xy_loss,wh_loss=wh_loss,no_object_loss=no_object_loss,object_loss=object_loss,vanilla_anchor=vanilla_anchor)

    # Load weights
    model.load_weights(weights_path, model.get_start_weight_dim())
    model.to(device, non_blocking=True)

    # detection
    BB_list = single_img_detect(target_img,model,device,conf_thres,nms_thres)
    return BB_list

def single_img_detect(target_img,model,device,conf_thres,nms_thres):

    img = target_img
    w, h = img.size
    new_width, new_height = model.img_size()
    pad_h, pad_w, ratio = calculate_padding(h, w, new_height, new_width)
    img = torchvision.transforms.functional.pad(img, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
    img = torchvision.transforms.functional.resize(img, (new_height, new_width))
    BB_list = []
    bw = model.get_bw()
    if bw:
        img = torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)

    img = torchvision.transforms.functional.to_tensor(img)
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        model.eval()
        img = img.to(device, non_blocking=True)
        # output,first_layer,second_layer,third_layer = model(img)
        output = model(img)


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
        img_with_boxes = target_img
        draw = ImageDraw.Draw(img_with_boxes)
        w, h = img_with_boxes.size

        for i in range(len(main_box_corner)):
            x0 = main_box_corner[i, 0].to('cpu').item() / ratio - pad_w
            y0 = main_box_corner[i, 1].to('cpu').item() / ratio - pad_h
            x1 = main_box_corner[i, 2].to('cpu').item() / ratio - pad_w
            y1 = main_box_corner[i, 3].to('cpu').item() / ratio - pad_h 
            # draw.rectangle((x0, y0, x1, y1), outline="red")
            # print("BB ", i, "| x = ", x0, "y = ", y0, "w = ", x1 - x0, "h = ", y1 - y0, "probability = ", probabilities_nms[i].item())
            BB = [round(x0), round(y0), round(y1 - y0), round(x1 - x0)]  # x, y, h, w
            BB_list.append(BB)
        #img_with_boxes.save('detected_cones.jpg')
        return BB_list




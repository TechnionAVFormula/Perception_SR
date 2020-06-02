import os
import ast
import cv2
import numpy as np
from PIL import Image
#from detect import get_BB_from_img
from config import CONFIG
from config import ConfigEnum
from PIL import Image, ImageDraw, ImageFont

if (CONFIG  == ConfigEnum.REAL_TIME) or (CONFIG == ConfigEnum.COGNATA_SIMULATION):
    from pyFormulaClient import messages
elif ( CONFIG == ConfigEnum.LOCAL_TEST):
    from pyFormulaClientNoNvidia import messages
else:
    raise NameError('User Should Choose Configuration from config.py')


def get_cones_from_camera(width, height, pixels):
    # convert from bit representation to RGB + depth format
    img_RGB, img_depth = convert_img_bits_to_RGBD(width, height, pixels)
    # set NN parameters
    weights_path = 'outputs/february-2020-experiments/yolo_baseline/9.weights'
    model_cfg = 'model_cfg/yolo_baseline.cfg'

    # Detect cones in input image via YOLO
    BB_list = get_BB_from_img(img_RGB,weights_path,model_cfg,conf_thres = 0.8,nms_thres = 0.25,xy_loss = 2,wh_loss = 1.6,no_object_loss = 25,object_loss=0.1,vanilla_anchor = False)

    # classify detected cone to types
    for BB in BB_list:
        cone_color = predict_cone_color(img_RGB,BB)
        cone_depth = predict_cone_depth(img_depth,BB)
        # BB = [x,y,h,w,type,depth]
        BB.append(cone_color)
        BB.append(cone_depth)
    # BB_list = [BB_1,BB_2,...,BB_N]
    return img_depth, BB_list

def convert_img_bits_to_RGBD(width, height, pixels):
    # convert bit format to RGB
    img_RGB = Image.frombytes("RGB", (width, height), pixels, 'raw', 'RGBX', 0,-1)
    # convert bit format to depth channel
    CHANNEL_COUNT = 4
    frames = np.frombuffer(pixels, dtype=np.uint8)
    deinterleaved = [frames[idx::CHANNEL_COUNT] for idx in range(CHANNEL_COUNT)]
    img_depth = np.reshape(deinterleaved[3], (width, height))

    return img_RGB, img_depth

def cut_cones_from_img(target_img, BB):
    [x, y, h, w] = BB
    crop_img = target_img.crop((x,y,x+w,y+h))
    return crop_img

def predict_cone_color(target_img, BB):

    frame = cut_cones_from_img(target_img, BB)
    frame = np.array(frame) # convert from PIL to cv
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    imgB = cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2HSV)
    imgY = cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2HSV)

    blueLow = np.array([90, 50, 50])
    blueHigh = np.array([150, 255, 255])
    yellowLow = np.array([20, 100, 100])
    yellowHigh = np.array([40, 255, 255])

    maskBlue = cv2.inRange(imgB, blueLow, blueHigh)
    maskBlue2 = cv2.dilate(maskBlue, np.ones((5, 5), np.uint8))
    maskBlue3 = cv2.erode(maskBlue2, np.ones((5, 5), np.uint8))
    maskYellow = cv2.inRange(imgY, yellowLow, yellowHigh)
    maskYellow2 = cv2.dilate(maskYellow, np.ones((5, 5), np.uint8))
    maskYellow3 = cv2.erode(maskYellow2, np.ones((5, 5), np.uint8))
    outputBlue = cv2.bitwise_and(imgB, imgB, mask=maskBlue3) ## TODO change to conv function
    outputYellow = cv2.bitwise_and(imgY, imgY, mask=maskYellow3) ## TODO change to conv function

    tempBlue = cv2.cvtColor(outputBlue, cv2.COLOR_BGR2GRAY)
    tempYellow = cv2.cvtColor(outputYellow, cv2.COLOR_BGR2GRAY)
    edgedBlue = cv2.Canny(tempBlue, 30, 300)
    edgedYellow = cv2.Canny(tempYellow, 30, 300)
    

    final_frame = cv2.hconcat((frame, outputYellow, outputBlue))
    cv2.imshow("final_frame", final_frame)

    n_white_pix_yellow = np.sum(yellow_mask == 255)
    n_white_pix_blue = np.sum(blue_mask == 255)
    n_white_pix = [n_white_pix_yellow, n_white_pix_blue]
    # print('Number of white pixels: Yellow_mask = ', n_white_pix_yellow, " | Blue_mask = ", n_white_pix_blue, " | Orange_mask = ", n_white_pix_orange)
    max_value = max(n_white_pix)
    max_idx = n_white_pix.index(max_value)

    if max_idx == 0: # cone is yellow
        return messages.perception.Yellow
    elif max_idx == 1: # cone is blue
        return messages.perception.Blue

def predict_cone_depth(img_depth, BB):
    frame = img_depth
    array = np.array(frame)
    max_pixel_value = array.max()
    return max_pixel_value

def trasform_img_cones_to_xyz(img_cones, img_depth, h_fov, v_fov, width, height):
    # get BB in image plain (img_cones) and transform it to xyz coordinates of camera (xyz_cones)

    # choose single representative point in each BB:
    img_cone_points = []  # list of (x,y,type) BB representative point in img plain
    for img_cone in img_cones:
        img_cone_points.append(get_BB_img_point(img_cone))

    # extract xyz coordinates of each cone
    xyz_cones = []  # list of (X,Y,Z,type) in ENU coordinate system (X - right, Y-forward, Z-upward)
    for img_cone_point in img_cone_points:
        #img_depth_px = img_depth.load()
        img_cone_point_depth = img_depth[img_cone_point[0:2]]  # specific point depth value
        xyz_cones.append(trasform_img_point_to_xyz(img_cone_point,img_cone_point_depth,h_fov,v_fov,width,height))
        # insert cone type to xyz_cones:
        xyz_cones[-1].append(img_cone_point[-1])

    return xyz_cones

def trasform_img_point_to_xyz(img_point, img_depth, h_fov, v_fov, width, height):
    # extract parameters
    u = img_point[0]
    v = img_point[1]
    alpha_h = (180 - h_fov)/2  # [deg]
    alpha_v = (180 - v_fov)/2  # [deg]
    # calculating gammas:
    gamma_h = alpha_h + (1-u / width) * h_fov  # [deg]
    gamma_v = alpha_v + (v / height) * v_fov  # [deg]
    # calculating X,Y,Z in ENU coordinate system (X - right, Y-forward, Z-upward)
    Y = img_depth
    X = img_depth / np.tan(gamma_h * np.pi / 180)
    Z = img_depth / np.tan(gamma_v * np.pi / 180)

    return [X, Y, Z]

def get_BB_img_point(img_cone):
    # return representative point in BB (center of BB at the moment)
    x = img_cone[0]
    y = img_cone[1]
    w = img_cone[2]
    h = img_cone[3]
    type = img_cone[4]
    return int(x+w/2), int(y+h/2), type

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
        type = type_map[raw_type-1]
        text = f"({i}) {type}"
        x1 = x0 + w
        y1 = y0 + h
        # draw BB + indicative text
        draw.rectangle((x0, y0, x1, y1), outline=type)
        w_text, h_text = font.getsize(text)
        draw.rectangle((x0, y0-h_text, x0 + w_text, y0), fill=type)
        draw.text((x0, y0-h_text),text , fill=(0, 0, 0, 128))

    return img_with_boxes

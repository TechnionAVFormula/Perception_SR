import os
import ast
import cv2
import numpy as np
from PIL import Image
from detect import get_BB_from_img
from perception_functions import predict_cone_color, predict_cone_depth, trasform_img_cones_to_xyz, draw_results_on_image
from config import CONFIG
from config import ConfigEnum
import pandas as pd
if (CONFIG  == ConfigEnum.REAL_TIME) or (CONFIG == ConfigEnum.COGNATA_SIMULATION):
    from pyFormulaClient import messages
elif ( CONFIG == ConfigEnum.LOCAL_TEST):
    from pyFormulaClientNoNvidia import messages
else:
    raise NameError('User Should Choose Configuration from config.py')

# parameters
df = pd.DataFrame([],columns=['u', 'v', 'h', 'w', 'X', 'Y', 'Z', 'type', 'frame'])
in_ind = 0
h_fov = 50  # [deg]
v_fov = 30  # [deg]
frame_number = 85  # update this number according to the video
# mapping from system runner cone type representation to string representation
type_map = [' '] * 3
type_map[messages.perception.Yellow - 1] = 'yellow'
type_map[messages.perception.Blue - 1] = 'blue'
type_map[messages.perception.Orange - 1] = 'orange'

# set NN parameters
weights_path = 'outputs/february-2020-experiments/yolo_baseline/9.weights'
model_cfg = 'model_cfg/yolo_baseline.cfg'

for i in range(1, frame_number+1):
    # open image from the image sequence
    img_RGB = Image.open('videos/big_lap_ordered_cones_1_left/big_lap_ordered_cones_1_left{:03n}0.jpg'.format(i)).convert('RGB')
    #img_depth = Image.open('simulation data/four_cones_depth.png')
    #img_depth = img_depth.load()
    width, height = img_RGB.width, img_RGB.height
    img_depth = np.random.randint(0,1000,(width, height))

    # Detect cones in input image via YOLO:
    # get image cones: BB_list=[[x,y,h,w,type,depth],[x,y,h,w,type,depth],....]
    # x,y - left top bounding box position in image plain
    # w, h - width and height of bounding box in pixels
    # type - cone color: 'B' - blue, 'Y' - yellow, 'O' - orange
    # depth - nominal depth value
    BB_list = get_BB_from_img(img_RGB,weights_path,model_cfg,conf_thres = 0.8,nms_thres = 0.25,xy_loss = 2,wh_loss = 1.6,no_object_loss = 25,object_loss=0.1,vanilla_anchor = False)
    # classify detected cone to types
    for BB in BB_list:
        cone_color = predict_cone_color(img_RGB, BB)
        cone_depth = predict_cone_depth(img_depth, BB)
        # BB = [x,y,w,h,type,depth]
        BB.append(cone_color)
        BB.append(cone_depth)

    # transformation from image plain to cartesian coordinate system
    # xyz_cones = [(X, Y, Z, type), (X, Y, Z, type), ....]
    # X,Y,Z - in ENU coordinate system (X - right, Y-forward, Z-upward)
    # type - cone color: 'B' - blue, 'Y' - yellow, 'O' - orange
    img_cones = BB_list
    xyz_cones = trasform_img_cones_to_xyz(img_cones, img_depth, h_fov, v_fov, width, height)

    out_ind = in_ind + len(BB_list)
    # fill in the dataframe:
    for BB, xyz_cone in zip(BB_list, xyz_cones):
        u = BB[0]
        v = BB[1]
        h = BB[2]
        w = BB[3]
        X = round(xyz_cone[0])
        Y = round(xyz_cone[1])
        Z = round(xyz_cone[2])
        type = type_map[xyz_cone[3]-1]
        df_row = pd.DataFrame(data=np.array([u, v, h, w, X, Y, Z, type, i]).reshape((1, 9)),
                              columns=['u', 'v', 'h', 'w', 'X', 'Y', 'Z', 'type', 'frame'])
        df = df.append(df_row, ignore_index=True)
    print(f'\rfinished processing: {i} / {frame_number}')

df.to_csv('videos/in5/in5_results.csv')
print(f'done!')

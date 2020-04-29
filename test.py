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
from timeit import default_timer as timer

def main():
    # convert from bit representation to RGB + depth format
    img_RGB = Image.open('simulation data/four_cones_raw.jpg').convert('RGB')
    img_depth = Image.open('simulation data/four_cones_depth.png')
    img_depth = img_depth.load()
    width, height = img_RGB.width, img_RGB.height
    h_fov = 50  # [deg]
    v_fov = 30  # [deg]

    # mapping from system runner cone type representation to string representation
    type_map = [' ']*3
    type_map[messages.perception.Yellow-1] ='yellow'
    type_map[messages.perception.Blue-1] ='blue'
    type_map[messages.perception.Orange-1] ='orange'

    # set NN parameters
    weights_path = 'outputs/february-2020-experiments/yolo_baseline/9.weights'
    model_cfg = 'model_cfg/yolo_baseline.cfg'

    # Detect cones in input image via YOLO:
    # get image cones: BB_list=[[x,y,h,w,type,depth],[x,y,h,w,type,depth],....]
    # x,y - left top bounding box position in image plain
    # w, h - width and height of bounding box in pixels
    # type - cone color: 'B' - blue, 'Y' - yellow, 'O' - orange
    # depth - nominal depth value
    start_detection = timer()
    BB_list = get_BB_from_img(img_RGB,weights_path,model_cfg,conf_thres = 0.8,nms_thres = 0.25,xy_loss = 2,wh_loss = 1.6,no_object_loss = 25,object_loss=0.1,vanilla_anchor = False)
    end_detection = timer()
    # classify detected cone to types
    start_classification = timer()
    for BB in BB_list:
        cone_color = predict_cone_color(img_RGB, BB)
        cone_depth = predict_cone_depth(img_depth, BB)
        # BB = [x,y,w,h,type,depth]
        BB.append(cone_color)
        BB.append(cone_depth)
    end_classification = timer()
    # print BB results
    print("Bounding box list in image plain:")
    for i, BB in enumerate(BB_list):
        print(f"({i}) x = {BB[0]}, y = {BB[1]}, h = {BB[2]}, w = {BB[3]}, type = {type_map[BB[4]-1]}, depth = {BB[5]}")

    # draw results on image
    img_with_boxes = draw_results_on_image(img_RGB, BB_list, type_map)
    img_with_boxes.save('detected_cones.jpg')

    # transformation from image plain to cartesian coordinate system
    # xyz_cones = [(X, Y, Z, type), (X, Y, Z, type), ....]
    # X,Y,Z - in ENU coordinate system (X - right, Y-forward, Z-upward)
    # type - cone color: 'B' - blue, 'Y' - yellow, 'O' - orange
    img_cones = BB_list
    start_xyz = timer()
    xyz_cones = trasform_img_cones_to_xyz(img_cones, img_depth, h_fov, v_fov, width, height)
    end_xyz = timer()
    # print XYZ results
    print("Cones X,Y,Z list in ENU coordinate system (X - right, Y - forward, Z - upward):")
    for i, xyz_cone in enumerate(xyz_cones):
        print(f"({i}) X = {int(xyz_cone[0])}, Y = {int(xyz_cone[1])}, Z = {int(xyz_cone[2])}, type = {type_map[xyz_cone[3]-1]}")

    # print timing analysis:
    print(f'Time duration:\n'
          f'detection: {round((end_detection-start_detection)*1000)} [ms]\n'
          f'classification: {round((end_classification-start_classification)*1000)} [ms]\n'
          f'xyz extraction: {round((end_xyz-start_xyz)*1000)} [ms]')

    # convert detection results to pandas dataframe format
    img_df = pd.DataFrame()
    img_df['u'] = [BB[0] for BB in BB_list]
    img_df['v'] = [BB[1] for BB in BB_list]
    img_df['h'] = [BB[2] for BB in BB_list]
    img_df['w'] = [BB[3] for BB in BB_list]
    img_df['X'] = [round(xyz_cone[0]) for xyz_cone in xyz_cones]
    img_df['Y'] = [round(xyz_cone[1]) for xyz_cone in xyz_cones]
    img_df['Z'] = [round(xyz_cone[2]) for xyz_cone in xyz_cones]
    img_df['type'] = [type_map[xyz_cone[3]-1] for xyz_cone in xyz_cones]
    img_df.to_csv('simulation data/detection_results.csv')
    return

if __name__ == "__main__":
    main()


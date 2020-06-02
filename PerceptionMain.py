from PerceptionClient import PerceptionClient
# from perception_functions import get_cones_from_camera
from perception_functions import cones_detection
from geometry import trasform_img_cones_to_xyz


import time
import signal
import os
import sys
import math
import random
import logging
import logging.handlers
import timeit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision
from models import Darknet

from PIL import Image, ImageDraw, ImageFont

from google.protobuf import json_format

from config import CONFIG, SAVE_RUN_DIR
from config import ConfigEnum

if (CONFIG  == ConfigEnum.REAL_TIME) or (CONFIG == ConfigEnum.COGNATA_SIMULATION):
    from pyFormulaClient import messages
    from pyFormulaClient import MessageDeque
elif ( CONFIG == ConfigEnum.LOCAL_TEST):
    from pyFormulaClientNoNvidia import messages
    from pyFormulaClientNoNvidia import MessageDeque
else:
    raise NameError('User Should Choose Configuration from config.py')
from timeit import default_timer as timer


class Perception:
    def __init__(self, weights_path, model_cfg, conf_thres = 0.8, nms_thres = 0.25, xy_loss = 2, wh_loss = 1.6, no_object_loss = 25, object_loss=0.1, vanilla_anchor = False):
        # The sensors.messages can be created using the create_sensors_file.py
        # PerceptionClient(<path to read messages from>, <path to write sent messages to>)
        self._client = PerceptionClient()
        self._running_id = 1
        self.message_timeout = 0.01

        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        logging.info("Running with %s", self.device)
        random.seed(0)
        torch.manual_seed(0)
        if cuda:
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
        self.model = Darknet(config_path=model_cfg,xy_loss=xy_loss,wh_loss=wh_loss,no_object_loss=no_object_loss,object_loss=object_loss,vanilla_anchor=vanilla_anchor)
   
        # Load weights
        self.model.load_weights(weights_path, self.model.get_start_weight_dim())
        self.model.to(self.device, non_blocking=True)
        

    def start(self):
        self._client.connect(1)
        if CONFIG == ConfigEnum.LOCAL_TEST:
            self._client.set_read_delay(0.05) # Sets the delay between reading new messages from sensors.messages
        self._client.start()

    def stop(self):
        if self._client.is_alive():
            self._client.stop()
            self._client.join()
 
    def process_camera_message(self, camera_msg, depth_camera_msg):
        logging.info("Started processing camera message id: %d, depth camera message id: %d", 
            camera_msg.header.id, depth_camera_msg.header.id)

        camera_data = messages.sensors.CameraSensor()
        camera_msg.data.Unpack(camera_data)
        depth_camera_data = messages.sensors.DepthCameraSensor()
        depth_camera_msg.data.Unpack(depth_camera_data)
        # Camera data has the following properties: width, height, pixels, h_fov, v_fov
        # print(f"Got camera width: {camera_data.width}, height: {camera_data.height}")

        # if SAVE_RUN_DIR is not None:
        #     with open(os.path.join(SAVE_RUN_DIR, f"{camera_data.frame_number}_camera_msg_{camera_msg.header.id}.bin"), 'wb') as f:
        #         f.write(camera_msg.SerializeToString())

        #     with open(os.path.join(SAVE_RUN_DIR, f"{depth_camera_data.frame_number}_depth_camera_msg_{depth_camera_msg.header.id}.bin"), 'wb') as f:
        #         f.write(depth_camera_msg.SerializeToString())

        logging.info("Processing camera message frame: %d, depth camera message frame: %d", 
            camera_data.frame_number, depth_camera_data.frame_number)

        # Create the new cone map and append to it all of the recognized cones from the image
        cone_map = messages.perception.ConeMap()

        # Get image cones: img_cones=[[x,y,h,w,type,depth],[x,y,h,w,type,depth],....]
        # x,y - left top bounding box position in image plain
        # w, h - width and height of bounding box in pixels
        # type - cone color: 'B' - blue, 'Y' - yellow, 'O' - orange
        # depth - nominal depth value

        # # set NN parameters     ################## need to set while creating peception object ##################
        # weights_path = 'weights/YOLOv3_1.weights' 
        # model_cfg = 'model_cfg/yolo_baseline.cfg'

        t1 = timer()
        # img_cones = get_cones_from_camera(camera_data.width, camera_data.height, camera_data.pixels, weights_path, model_cfg)
        
        # img_cones is a list of dict: [['u', 'v', 'h', 'w', 'pr', 'type'], ['u', 'v', 'h', 'w', 'pr', 'type], ....]
        img_cones = cones_detection(camera_data.width, camera_data.height, camera_data.pixels,
                                    self.model, self.device, self.conf_thres, self.nms_thres)

        t2 = timer()
        logging.info("'get_cones_from_camera' took %d [ms]", round((t2-t1)*1000))

        # transformation from image plain to cartesian coordinate system
        # xyz_cones is list of lists [[X, Y, Z, type], [X, Y, Z, type], ....]
        # X,Y,Z - in ENU coordinate system (X - right, Y-forward, Z-upward)
        # type - cone color: 'B' - blue, 'Y' - yellow, 'O' - orange
        camera_pos = camera_data.config.sensor_position
        if len(img_cones) == 0:
            return None

        xyz_cones = trasform_img_cones_to_xyz(img_cones, camera_data.width, camera_data.height,
                                            depth_camera_data.config.data_type, depth_camera_data.pixels,
                                            camera_data.config.hfov, camera_data.config.vfov, camera_pos)

        for index, xyz_cone in enumerate(xyz_cones):
            #   Create new cone and set its properties
            cone = messages.perception.Cone()
            cone.cone_id = index
            cone.type = xyz_cone[3]
            cone.x = xyz_cone[0]
            cone.y = xyz_cone[1]
            cone.z = xyz_cone[2]
            cone_map.cones.append(cone)  # append the new cone to the cone map

        logging.info("Finished processing camera message %d", camera_msg.header.id)
        return cone_map

    def process_server_message(self, server_messages):
        if server_messages.data.Is(messages.server.ExitMessage.DESCRIPTOR):
            return True

        return False

    def send_message2state(self, msg_id, cone_map):
        msg = messages.common.Message()
        msg.header.id = msg_id
        msg.data.Pack(cone_map)
        self._client.send_message(msg)

    def check_for_server_messages(self):
        try:
            server_msg = self._client.pop_server_message()
            if server_msg is not None:
                if self.process_server_message(server_msg):
                    return  
        except MessageDeque.NoFormulaMessages:
            pass
                
    def check_for_camera_messages(self):
        try:
            # In the future can be replaced with getting the camera directly
            camera_msg, depth_camera_msg = self._client.get_camera_message(timeout=self.message_timeout) 
            cone_map = self.process_camera_message(camera_msg, depth_camera_msg)
            if cone_map is None:
                logging.info("No cones detected")
                return
                
            logging.info("Outputing cone map: %s", json_format.MessageToJson(cone_map))

            self.send_message2state(camera_msg.header.id, cone_map)    
        except MessageDeque.NoFormulaMessages:
            pass

    def debug_process_ground_truth_message(self, ground_truth_message):
        ground_truth_data = messages.ground_truth.GroundTruth()
        ground_truth_message.data.Unpack(ground_truth_data)
            
        cone_map = messages.perception.ConeMap()
        for bb in ground_truth_data.perception_ground_truth.bbs:
            cone = messages.perception.Cone()
            cone.cone_id = bb.cone_id
            cone.type = bb.type
            cone.x = bb.position.x
            cone.y = bb.position.y
            cone.z = bb.position.z
            cone_map.cones.append(cone)  # append the new cone to the cone map
        
        return cone_map

                
    def check_for_ground_truth_messages(self):
        try:
            # In the future can be replaced with getting the camera directly
            ground_truth_message = self._client.get_ground_truth_message(timeout=self.message_timeout) 
            cone_map = self.debug_process_ground_truth_message(ground_truth_message)

            logging.info("Outputing cone map: %s", json_format.MessageToJson(cone_map))

            self.send_message2state(ground_truth_message.header.id, cone_map)    
        except MessageDeque.NoFormulaMessages:
            pass

    def run(self):
        while True:
            try:
                self.check_for_server_messages()
                self.check_for_camera_messages()
                # self.check_for_ground_truth_messages()
            except Exception:
                logging.warn("Got exception in loop", exc_info=True)
            
weights_path = 'outputs/february-2020-experiments/yolo_baseline/9.weights'
model_cfg = 'model_cfg/yolo_baseline.cfg'
perception = Perception(weights_path, model_cfg)

def stop_all_threads():
    print("Stopping threads")
    perception.stop()

def shutdown(a, b):
    print("Shutdown was called")
    stop_all_threads()
    exit(0)


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    file_handler = logging.handlers.WatchedFileHandler("perception.log", 'w')
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(ch)
    logger.addHandler(file_handler)

    logging.info("Initalized Perception")

    perception.start()
    perception.run()

    stop_all_threads()
    exit(0)

if __name__ == "__main__":
    for signame in ('SIGINT', 'SIGTERM'):
        signal.signal(getattr(signal, signame), shutdown)
    main()
from PerceptionAlgo.camera_image import CameraImage
from PerceptionAlgo.depth_camera_image import DepthImage
from PerceptionAlgo.yolov3_network import YoloV3Network
from PerceptionAlgo.geometry import img_cones_to_world_cones

from .perception_client import PerceptionClient

from .config import CONFIG, SAVE_RUN_DIR
from .config import ConfigEnum

if (CONFIG  == ConfigEnum.REAL_TIME) or (CONFIG == ConfigEnum.COGNATA_SIMULATION):
    from pyFormulaClient import messages
    from pyFormulaClient import MessageDeque
elif ( CONFIG == ConfigEnum.LOCAL_TEST):
    from pyFormulaClientNoNvidia import messages
    from pyFormulaClientNoNvidia import MessageDeque
else:
    raise NameError('User Should Choose Configuration from config.py')

import time
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

from PIL import Image, ImageDraw, ImageFont

from google.protobuf import json_format

from timeit import default_timer as timer


class PerceptionModuleRunner:
    def __init__(self, weights_path, model_cfg):
        # The sensors.messages can be created using the create_sensors_file.py
        # PerceptionClient(<path to read messages from>, <path to write sent messages to>)
        self._client = PerceptionClient()
        self._running_id = 1
        self.message_timeout = 0.01
        # Initializing the camera by creating the desired object
        # self.camera = ....

        # Initializing the cone detector method by creating the desired object
        self.cone_detector = YoloV3Network(weights_path, model_cfg)

        logging.info("Running with %s", self.cone_detector.device)
        

    def start(self):
        self._client.connect(1)
        if CONFIG == ConfigEnum.LOCAL_TEST:
            self._client.set_read_delay(0.05) # Sets the delay between reading new messages from sensors.messages
        self._client.start()

    def stop(self):
        if self._client.is_alive():
            self._client.stop()
            self._client.join()
 
    @staticmethod
    def world_cone_color_to_msg_color(color):
        if color == 'yellow':
            return messages.perception.Yellow
        elif color == 'blue':
            return messages.perception.Blue
            
    def process_camera_message(self, camera_msg, depth_camera_msg):
        logging.info("Started processing camera message id: %d, depth camera message id: %d", 
                    camera_msg.header.id, depth_camera_msg.header.id)

        camera_data = messages.sensors.CameraSensor() # Camera data has the following properties: width, height, pixels, h_fov, v_fov
        camera_msg.data.Unpack(camera_data)
        depth_camera_data = messages.sensors.DepthCameraSensor() # Depth Camera data has the following properties: ????????????
        depth_camera_msg.data.Unpack(depth_camera_data)
        # Creating an image object for camera image and deapth image
        camera_img = CameraImage(camera_data.pixels, camera_data.width, camera_data.height)
        deapth_img = DepthImage(depth_camera_data.pixels, depth_camera_data.width, depth_camera_data.height,
                                depth_camera_data.config.data_type, camera_data.config.hfov, 
                                camera_data.config.vfov, camera_data.config.sensor_position)
                                    ###### maybe need camera_data.width/height ????????




        logging.info("Processing camera message frame: %d, depth camera message frame: %d", 
                    camera_data.frame_number, depth_camera_data.frame_number)

        # Create the cone map message that will contain all the detected cones of the image
        cone_map = messages.perception.ConeMap()

        t1 = timer()
        # img_cones_list is a list of BoundingBoxCone objects
        img_cones_list = self.cone_detector.detect(camera_img)
        t2 = timer()
        logging.info("'get_cones_from_camera' took %d [ms]", round((t2-t1)*1000))

        
        if len(img_cones_list) == 0:
            return None

        # Transformation from image plain to 3D world cartesian coordinate system
        # cone_map is list of WorldCone objects
        # x,y - in ENU coordinate system (X - right, Y-forward, Z-upward)
        # type - cone color messege: blue, yellow, orange
        ###### Where to put this function - utils ???? because it not belong to any class ####################### 
        world_cones = img_cones_to_world_cones(img_cones_list, deapth_img)

        # Preparing the cone map message
        for cone in world_cones:
            #   Create new cone and set its properties
            map_cone = messages.perception.Cone()
            map_cone.cone_id = cone.id
            map_cone.type = PerceptionModuleRunner.world_cone_color_to_msg_color(cone.color)
            map_cone.x = cone.x
            map_cone.y = cone.y
            map_cone.confidence = cone.pr
            # map_cone.z = 
            cone_map.cones.append(map_cone)  # append the new cone to the cone map

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

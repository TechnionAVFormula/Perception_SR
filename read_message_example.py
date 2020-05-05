from pyFormulaClientNoNvidia import messages
from PIL import Image
from perception_functions import get_cones_from_camera, trasform_img_cones_to_xyz, draw_results_on_image
from google.protobuf import json_format

# Parameters
type_map = [' ']*3
type_map[messages.perception.Yellow-1] ='yellow'
type_map[messages.perception.Blue-1] ='blue'
type_map[messages.perception.Orange-1] ='orange'

# Open message file: {msg frame}_camera_msg_{msg id}.bin
with open('Run_1_05_20/125_camera_msg_624.bin', 'rb') as f:
    buffer = f.read()
    camera_msg = messages.common.Message()
    camera_msg.ParseFromString(buffer)
with open('Run_1_05_20/125_depth_camera_msg_623.bin', 'rb') as f:
    buffer = f.read()
    depth_camera_msg = messages.common.Message()
    depth_camera_msg.ParseFromString(buffer)
with open('Run_1_05_20/125_ground_truth_625.bin', 'rb') as f:
    buffer = f.read()
    ground_truth_msg = messages.common.Message()
    ground_truth_msg.ParseFromString(buffer)

# Extract data from camera
camera_data = messages.sensors.CameraSensor()
camera_msg.data.Unpack(camera_data)
depth_camera_data = messages.sensors.DepthCameraSensor()
depth_camera_msg.data.Unpack(depth_camera_data)
ground_truth_data = messages.ground_truth.GroundTruth()
ground_truth_msg.data.Unpack(ground_truth_data)

# Process camera data
img_cones = get_cones_from_camera(camera_data.width, camera_data.height, camera_data.pixels)
xyz_cones = trasform_img_cones_to_xyz(img_cones, depth_camera_data.config.data_type, depth_camera_data.pixels, camera_data.config.hfov, camera_data.config.vfov, camera_data.width, camera_data.height)

# Print detection results
print("Bounding box list in image plain:")
for i, BB in enumerate(img_cones):
    print(f"({i}) u = {BB[0]}, v = {BB[1]}, h = {BB[2]}, w = {BB[3]}, type = {type_map[BB[4]-1]}")
print("Cones X,Y,Z list in ENU coordinate system (X - right, Y - forward, Z - upward):")
for i, xyz_cone in enumerate(xyz_cones):
    print(f"({i}) X = {int(xyz_cone[0])}, Y = {int(xyz_cone[1])}, Z = {int(xyz_cone[2])}, type = {type_map[xyz_cone[3] - 1]}")
# Print GT results
print(json_format.MessageToJson(ground_truth_data))

# Export captured images
img_RGB = Image.frombytes("RGB", (camera_data.width, camera_data.height), camera_data.pixels, 'raw', 'RGBX', 0,-1)
img_RGB.save('125_camera_msg_4.png')
dc_img = Image.frombytes("I;16", (depth_camera_data.width, depth_camera_data.height), depth_camera_data.pixels)
dc_img.save('125_depth_camera_msg_3.png')


# draw detection results on image and export
img_RGB_boxes = draw_results_on_image(img_RGB, img_cones, type_map)
img_RGB_boxes.save('125_RGB_detected_cones.jpg')

# close images
img_RGB.close()
dc_img.close()
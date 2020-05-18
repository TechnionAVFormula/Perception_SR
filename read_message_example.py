from pyFormulaClientNoNvidia import messages
from PIL import Image
from perception_functions import get_cones_from_camera, draw_results_on_image
from geometry import trasform_img_cones_to_xyz, compare_XYZ_to_GT
from google.protobuf import json_format
import ast
import csv

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

camera_pos = camera_data.config.sensor_position
camera_pos.z = 1.2  # an error occurred which sets z=0 in these sample messages
camera_pos_depth = depth_camera_data.config.sensor_position

# Extract ground truth data
xyz_cones_GT = []
for bb in ground_truth_data.perception_ground_truth.bbs:
    xyz_cones_GT.append([bb.position.x, bb.position.y, bb.position.z, bb.type])

# Read detected cones from file
img_cones = []
with open('Run_1_05_20/125_detection.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentLine = line[:-1]
        # convert line string to list
        currentLine = ast.literal_eval(currentLine)
        # add item to the list
        img_cones.append(currentLine)

# Process camera data
# img_cones = get_cones_from_camera(camera_data.width, camera_data.height, camera_data.pixels)
xyz_cones = trasform_img_cones_to_xyz(img_cones, camera_data.width, camera_data.height,
                                      depth_camera_data.config.data_type, depth_camera_data.pixels,
                                      camera_data.config.hfov, camera_data.config.vfov, camera_pos)
# Write detected cones to file:
# with open('Run_1_05_20/428_detection.txt', 'w') as f:
# #    for item in img_cones:
# #        f.write("%s\n" % item)

# compare ground truth to detection:
indices = compare_XYZ_to_GT(xyz_cones, xyz_cones_GT)

# save results to csv
with open('Run_1_05_20/125_detection_vs_ground_truth.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([f"detected: {len(xyz_cones)} / {len(xyz_cones_GT)}",
                     "units: [meters]", "X - forward","Y - Right", "Z - Up"])
    writer.writerow(["cone #", "Src", "X", "Y", "Z"])
    for i, xyz_cone in enumerate(xyz_cones):
        writer.writerow([i,"detected", round(xyz_cone[0],2), round(xyz_cone[1],2), round(xyz_cone[2],2)])
        writer.writerow([" ", "GT", round(xyz_cones_GT[indices[i]][0], 2), round(xyz_cones_GT[indices[i]][1], 2), round(xyz_cones_GT[indices[i]][2], 2)])

# Print detection results
print("Bounding box list in image plain:")
for i, BB in enumerate(img_cones):
    print(f"({i}) u = {BB[0]}, v = {BB[1]}, h = {BB[2]}, w = {BB[3]}, type = {type_map[BB[4]-1]}")
print("Cones X,Y,Z list in cognata coordinate system (X - right, Y - forward, Z - upward):")
for i, xyz_cone in enumerate(xyz_cones):
    print(f"=====================Cone ({i})==========================")
    print("---------------------Detected--------------------------")
    print(f"X = {round(xyz_cone[0],2)}, Y = {round(xyz_cone[1],2)}, Z = {round(xyz_cone[2],2)}, type = {type_map[xyz_cone[3] - 1]}")
    print("--------------------closest GT-------------------------")
    print(f"X = {round(xyz_cones_GT[indices[i]][0], 2)}, Y = {round(xyz_cones_GT[indices[i]][1], 2)}, Z = {round(xyz_cones_GT[indices[i]][2], 2)}, type = {type_map[xyz_cones_GT[indices[i]][3] - 1]}\n")

# Export captured images
img_RGB = Image.frombytes("RGB", (camera_data.width, camera_data.height), camera_data.pixels, 'raw', 'RGBX', 0,-1)
img_RGB.save('Run_1_05_20/125_camera_msg_624.png')
dc_img = Image.frombytes("I;16", (depth_camera_data.width, depth_camera_data.height), depth_camera_data.pixels)
dc_img.save('Run_1_05_20/125_depth_camera_msg_623.png')


# draw detection results on image and export
img_RGB_boxes = draw_results_on_image(img_RGB, img_cones, type_map)
img_RGB_boxes.save('Run_1_05_20/125_RGB_detected_cones.jpg')

# close images
img_RGB.close()
dc_img.close()
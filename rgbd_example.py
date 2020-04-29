import cv2
import numpy as np
from geometry import *
import pptk

several_points_only = True

hFOV = 50*np.pi/180

rgb_img = cv2.imread('Simulation data/four_cones_raw.jpg')
d_img = cv2.imread('Simulation data/four_cones_depth.png', cv2.IMREAD_ANYDEPTH)  # the flag is important, this is a 16-bit image

assert (rgb_img.shape[0:2] == d_img.shape[0:2])
height = rgb_img.shape[0]  # horizontal size (actually vertical (Gilad))
width = rgb_img.shape[1]
f = width/(2*np.tan(hFOV/2))  # 1/2w        (hFOV)
                              # ----  = tan (----)
                              #  f          (  2 )
vFOV = np.arctan(270/f)
cx = width/2
cy = height/2 + 7 # horizon is 7 pixels below the centerline

K = np.array([[f,0,cx],
              [0,f,cy],
              [0,0, 1]], dtype=np.float)

angle = - np.pi/2.0

Rinv = np.array([[1,             0,              0],
                 [0, np.cos(angle), -np.sin(angle)],
                 [0, np.sin(angle),  np.cos(angle)]], dtype=np.float)

tinv = np.array([0,0,120], dtype=np.float)

if several_points_only:
    image_points = np.asarray([[465.0, 390.0],
                               [695.0, 388.0],
                               [745.5, 405.0],
                               [829.0, 463.0]], dtype=np.float64)
else:
    # X, Y = np.mgrid[671:861, 367:495]
    X, Y = np.mgrid[300:860, 350:height]
    image_points = np.vstack((X.ravel(), Y.ravel())).T

index_x = image_points[:,0].astype(np.int)
index_y = image_points[:,1].astype(np.int)

depths = d_img[index_y, index_x]

positions = world_XYZ_from_uvd(image_points, depths=depths, K=K, Rinv=Rinv, tinv=tinv)

if several_points_only:
    for i in range(len(image_points)):
        coords = tuple(image_points.astype(np.int)[i])
        coords_text = tuple(image_points.astype(np.int)[i] - np.array([100,0]))
        cv2.circle(rgb_img, coords, 2, (0, 0, 255), -1)
        cv2.putText(rgb_img, "Y=%.0fcm,X=%.0fcm,Z=%.0fcm" % (positions[i, 1], positions[i, 0], positions[i,2]),
                    coords_text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 255, 0))
    cv2.imshow('a', rgb_img)
    cv2.waitKey()
else:
    v = pptk.viewer(positions)
    colors = rgb_img[index_y, index_x]
    v.set(r=600, phi=-1.91440809, theta=0.1349903)
    v.set(point_size=0.5)
    v.attributes(colors / 255.)




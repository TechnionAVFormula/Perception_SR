import cv2
import numpy as np


def inverse_perspective(R, t):
    Ri = np.transpose(R)  # for a rotation matrix, inverse is the transpose
    ti = -Ri @ t
    return Ri, ti


def world_XY_from_uv_and_Z(points, K, R, t, Z):
    points_h = cv2.convertPointsToHomogeneous(points)  # just turns (u, v) into (u, v, 1)
    Ri = R.T
    Ki = np.linalg.inv(K)

    positions = np.zeros((len(points), 3), dtype=np.float64)

    obj_vector = Ri @ t
    for i in range(len(points)):
        uv = np.squeeze(points_h[i, :])
        img_vector = Ri @ Ki @ uv
        s = (Z + obj_vector[2])/img_vector[2]
        obj_pos = Ri @ (s * Ki @ uv - t)
        positions[i, :] = obj_pos
    return positions

def world_XYZ_from_uvd(points, depths, K, Rinv, tinv):
    Kinv = np.linalg.inv(K)
    uv1 = cv2.convertPointsToHomogeneous(points)[:,0,:]
    # s(u,v,1) = K(R(xyz)+t)
    # xyz = Rinv*(Kinv*s*(uv1)) + tinv
    image_vectors = np.multiply(uv1.T, [depths]) # depths is broadcast over the 3 rows of uv.T
    positions = (Rinv@Kinv@image_vectors).T + tinv  # tinv automatically broadcast to all matrix rows
    return positions


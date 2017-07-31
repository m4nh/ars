import cv2
import PyKDL
import roars.geometry.transformations as transformations
import numpy as np


def reproject3DPoints(world_points, camera=None, camera_pose=PyKDL.Frame()):
    cRvec, cTvec = transformations.KDLToCv(camera_pose)
    img_points, _ = cv2.projectPoints(
        world_points,
        cRvec,
        cTvec,
        camera.camera_matrix,
        camera.distortion_coefficients
    )
    img_points = np.reshape(img_points, (2))
    return img_points


def reproject3DPoint(x, y, z, camera=None, camera_pose=PyKDL.Frame()):
    obj_points = np.array([[x, y, z]], dtype="float32")
    cRvec, cTvec = transformations.KDLToCv(camera_pose)
    img_points, _ = cv2.projectPoints(
        obj_points,
        cRvec,
        cTvec,
        camera.camera_matrix,
        camera.distortion_coefficients
    )
    img_points = np.reshape(img_points, (2))
    return img_points

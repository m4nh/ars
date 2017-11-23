#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.cameras import CameraRGB
from roars.vision.arucoutils import MarkerDetector
from roars.vision.arp import ARP
from roars.vision.augmentereality import BoundingBoxFromSixPoints, VirtualObject
import roars.geometry.transformations as transformations

import roars.vision.cvutils as cvutils
import cv2
import numpy as np
import os
import glob
import sys
import shutil

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("rgb_dataset_to_roars")

scene_folder = node.setupParameter("scene_folder", "")
out_folder = node.setupParameter("out_folder", "")
zeros = node.setupParameter("zeros", 5)

basename = os.path.basename(scene_folder)

output_folder = os.path.join(out_folder, basename)
rgb_path = os.path.join(output_folder, "images")
depth_path = os.path.join(output_folder, "depth")
pose_path = os.path.join(output_folder, "robot_poses.txt")
extr_path = os.path.join(output_folder, "camera_extrinsics.txt")
intr_path = os.path.join(output_folder, "camera_intrinsics.txt")
try:
    os.makedirs(output_folder)
    os.makedirs(rgb_path)
    os.makedirs(depth_path)
except:
    pass

files = glob.glob(os.path.join(scene_folder, "*.*"))

rgb_files = []
depth_files = []
pose_files = []

for f in files:
    if 'color' in f:
        rgb_files.append(f)
    if 'depth' in f:
        depth_files.append(f)
    if 'pose' in f:
        pose_files.append(f)

rgb_files = sorted(rgb_files)
depth_files = sorted(depth_files)
pose_files = sorted(pose_files)

poses = []
for i in range(0, len(rgb_files)):
    image_name = str(i).zfill(zeros) + ".png"
    rgb_basename = os.path.basename(rgb_files[i])
    depth_basename = os.path.basename(depth_files[i])
    shutil.copy(rgb_files[i], os.path.join(rgb_path, image_name))
    shutil.copy(depth_files[i], os.path.join(depth_path, image_name))
    pose_raw = np.loadtxt(pose_files[i])
    pose = transformations.NumpyMatrixToKDL(pose_raw)
    pose_q = transformations.KDLtoNumpyVector(pose).ravel()
    print pose_q
    poses.append(pose_q)

poses = np.array(poses)
print poses.shape
np.savetxt(pose_path, poses)
np.savetxt(extr_path, np.array([0, 0, 0, 0, 0, 0, 1]))
np.savetxt(intr_path, np.array([640, 480, 570.3, 570.3, 320, 240, 0, 0, 0, 0]))

# img_folder = os.path.join(dataset_folder, "images")
# label_folder = os.path.join(dataset_folder, "labels")
# ids_folder = os.path.join(dataset_folder, "ids")

# if not os.path.exists(img_folder):
#     print("Invalud path '{}'".format(img_folder))
#     sys.exit(0)


# image_files = sorted(glob.glob(img_folder + "/*.jpg"))
# label_files = sorted(glob.glob(label_folder + "/*.txt"))
# id_files = sorted(glob.glob(ids_folder + "/*.txt"))

# ids = []
# for id_file in id_files:
#     f = open(id_file, 'r')
#     ids.append(f.readline().replace('\n', ''))
#     f.close()


# for id in ids:
#     print(id)

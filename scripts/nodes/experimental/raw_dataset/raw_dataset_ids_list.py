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

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("raw_dataset_ids_list")

dataset_folder = node.setupParameter("dataset_folder", "")

img_folder = os.path.join(dataset_folder, "images")
label_folder = os.path.join(dataset_folder, "labels")
ids_folder = os.path.join(dataset_folder, "ids")

if not os.path.exists(img_folder):
    print("Invalud path '{}'".format(img_folder))
    sys.exit(0)


image_files = sorted(glob.glob(img_folder + "/*.jpg"))
label_files = sorted(glob.glob(label_folder + "/*.txt"))
id_files = sorted(glob.glob(ids_folder + "/*.txt"))

ids = []
for id_file in id_files:
    f = open(id_file, 'r')
    ids.append(f.readline().replace('\n', ''))
    f.close()


for id in ids:
    print(id)

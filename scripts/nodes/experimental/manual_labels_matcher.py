#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.cameras import CameraRGB
from roars.vision.arucoutils import MarkerDetector
from roars.vision.augmentereality import VirtualObject
import roars.geometry.transformations as transformations
import cv2
import numpy as np
import os
import glob
from shutil import copyfile
import shutil
import collections


#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("manual_labels_matcher")

roars_dataset_folder = node.setupParameter(
    "roars_dataset_folder", "/home/daniele/Desktop/datasets/roars_2017/indust")

output_folder_test = node.setupParameter(
    "output_folder", "/home/daniele/Desktop/datasets/roars_manual_2017/manual_dataset_test")
output_folder_train = node.setupParameter(
    "output_folder", "/home/daniele/Desktop/datasets/roars_manual_2017/manual_dataset_train")

elements = []
for file in glob.glob(dataset_folder + "/*.txt"):
    name = os.path.basename(file).split(".")[0]
    img_folder = os.path.join(os.path.dirname(file), name + "_images")
    labels_folder = os.path.join(os.path.dirname(file), name + "_labels")
    f = open(file, 'r')

    element = {
        "file": file,
        "img_folder": img_folder,
        "labels_folder": labels_folder,
        "name": name,
        "ext": os.path.basename(file).split(".")[1],
        "images": sorted(glob.glob(img_folder + "/*.jpg")),
        "labels": sorted(glob.glob(labels_folder + "/*.txt")),
        "ids": f.readlines()
    }
    elements.append(element)


id_map = copyOutput(elements, output_folder_train, [
                    'indust_scene_4_top', 'indust_scene_4_dome', 'smartphone'], False)
id_map = copyOutput(elements, output_folder_test, [
                    'indust_scene_4_top', 'indust_scene_4_dome', 'smartphone'], True)

for k, v in id_map.iteritems():
    print(k, v)

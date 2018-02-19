#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.cameras import CameraRGB
from roars.vision.arucoutils import MarkerDetector
from roars.vision.arp import ARP
from roars.vision.augmentereality import BoundingBoxFromSixPoints, VirtualObject
from roars.datasets.datasetutils import RawDataset
import roars.geometry.transformations as transformations
from collections import OrderedDict
import roars.vision.cvutils as cvutils
import cv2
import numpy as np
import os
import glob
import sys
import shutil

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("raw_datasets_merge_all")

datasets_folder = node.setupParameter("datasets_folder", "")
output_folder = node.setupParameter("output_folder", "")
include_tags = node.setupParameter("include_tags", "gt_", array_type=str)
exclude_tags = node.setupParameter("exclude_tags", "4_", array_type=str)
debug = node.setupParameter("debug", True)
raw_dataset_folders = sorted(os.listdir(datasets_folder))


print("Include tags", include_tags, len(include_tags))

#⬢⬢⬢⬢⬢➤ Filter
folders = []
for d in os.listdir(datasets_folder):

    if any(w in d for w in include_tags):
        if not any(w in d for w in exclude_tags):

            folders.append(os.path.join(datasets_folder, d))
folders = sorted(folders)


#⬢⬢⬢⬢⬢➤ Merge data
data_list = []
for folder in folders:
    print("Parsing dataset", folder)
    dataset = RawDataset(folder)
    data_list.extend(dataset.data_list)


#⬢⬢⬢⬢⬢➤ Output
if len(output_folder) == 0 or debug:
    print("Output folder invalid!", output_folder)
    print len(data_list)
    sys.exit(0)
dataset = RawDataset(output_folder, create=True)

ZERO_PADDING_SIZE = 5
counter = 0
for data in data_list:

    counter_string = '{}'.format(str(counter).zfill(ZERO_PADDING_SIZE))

    img_file = os.path.join(dataset.img_folder, counter_string + ".jpg")
    label_file = os.path.join(dataset.label_folder, counter_string + ".txt")
    id_file = os.path.join(dataset.ids_folder, counter_string + ".txt")

    shutil.copyfile(data['image'], img_file)
    shutil.copyfile(data['id'], id_file)
    shutil.copyfile(data['label'], label_file)
    print data['id']
    counter = counter + 1


# data_list = []
# #⬢⬢⬢⬢⬢➤ Groups all data
# whole_data = {}
# for d in raw_dataset_folders:
#     dataset_path = os.path.join(datasets_folder, d)
#     dataset = RawDataset(dataset_path)
#     data_list.extend(dataset.data_list)


# dataset = RawDataset(output_folder, create=True)

# ZERO_PADDING_SIZE = 5
# counter = 0
# for data in data_list:

#     counter_string = '{}'.format(str(counter).zfill(ZERO_PADDING_SIZE))

#     img_file = os.path.join(dataset.img_folder, counter_string + ".jpg")
#     label_file = os.path.join(dataset.label_folder, counter_string + ".txt")
#     id_file = os.path.join(dataset.ids_folder, counter_string + ".txt")

#     shutil.copyfile(data['image'], img_file)
#     shutil.copyfile(data['id'], id_file)
#     shutil.copyfile(data['label'], label_file)

#     print data['id']

#     counter = counter + 1

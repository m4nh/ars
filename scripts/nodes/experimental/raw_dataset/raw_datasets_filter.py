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
node = RosNode("raw_datasets_filter")

datasets_folder = node.setupParameter("datasets_folder", "")
output_folder = node.setupParameter("output_folder", "")
filter_file = node.setupParameter("filter_file", "")
negative = node.setupParameter("negative", False)

filter_ids = open(filter_file, 'r').read().splitlines()

raw_dataset_folders = sorted(os.listdir(datasets_folder))

#⬢⬢⬢⬢⬢➤ Groups all data
whole_data = {}
for d in raw_dataset_folders:
    dataset_path = os.path.join(datasets_folder, d)
    dataset = RawDataset(dataset_path)
    for id, data in dataset.data_map.iteritems():
        print("Adding", id)
        whole_data[id] = data
print("whole data size", len(whole_data))

#⬢⬢⬢⬢⬢➤ Filter data
filtered_data = []

if negative:
    for id in filter_ids:
        filtered_data.append(whole_data[id])

else:
    for id, data in whole_data.iteritems():
        if id not in filter_ids:
            filtered_data.append(data)

print(len(filtered_data))

dataset = RawDataset(output_folder, create=True)
ZERO_PADDING_SIZE = 5
counter = 0
for data in filtered_data:

    counter_string = '{}'.format(str(counter).zfill(ZERO_PADDING_SIZE))

    img_file = os.path.join(dataset.img_folder, counter_string + ".jpg")
    label_file = os.path.join(dataset.label_folder, counter_string + ".txt")
    id_file = os.path.join(dataset.ids_folder, counter_string + ".txt")

    shutil.copyfile(data['image'], img_file)
    shutil.copyfile(data['id'], id_file)
    shutil.copyfile(data['label'], label_file)

    print counter_string, data['id']

    counter = counter + 1

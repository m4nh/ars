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
import random


def generateRandomSubset(oringal_set, test_percentage, validity_percentage=0.0):
    frames = list(oringal_set)
    random.shuffle(frames)

    all_count = len(frames)
    test_count = int(all_count * test_percentage)
    val_count = int(all_count * validity_percentage)
    train_count = all_count - test_count - val_count

    trains = frames[:train_count]
    remains = frames[train_count:]

    tests = remains[:test_count]
    vals = remains[test_count:]
    return {"train": trains, "test": tests, "val": vals}


#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("raw_datasets_merge_all")

datasets_folder = node.setupParameter("datasets_folder", "")
output_folder = node.setupParameter("output_folder", "")
test_percentage = node.setupParameter("test_percentage", 0.1)
val_percentage = node.setupParameter("val_percentage", 0.05)

if len(output_folder) == 0:
    print("Error in outputfolder")
    sys.exit(0)

os.mkdir(output_folder)

raw_dataset_folders = sorted(os.listdir(datasets_folder))

print raw_dataset_folders

data_list = []
#⬢⬢⬢⬢⬢➤ Groups all data
images_list = []
for d in raw_dataset_folders:
    dataset_path = os.path.join(datasets_folder, d)
    subfolders = os.listdir(dataset_path)
    print "##", dataset_path
    for s in subfolders:
        sub_path = os.path.join(dataset_path, s)
        images = glob.glob(os.path.join(sub_path, "*.jpg"))
        images_list.extend(images)
        print sub_path, len(images_list)

random.shuffle(images_list)

mapped_data = generateRandomSubset(
    images_list, test_percentage, val_percentage)


ZERO_PADDING_SIZE = 5
counter = 0
for k, data in mapped_data.iteritems():
    counter = 0
    sub_folder = os.path.join(output_folder, k)
    os.mkdir(sub_folder)

    for img_file in data:

        counter_string = '{}'.format(str(counter).zfill(ZERO_PADDING_SIZE))

        out_img_file = os.path.join(sub_folder, counter_string + ".jpg")
        shutil.copyfile(img_file, out_img_file)
        # label_file = os.path.join(
        #     dataset.label_folder, counter_string + ".txt")
        # id_file = os.path.join(dataset.ids_folder, counter_string + ".txt")

        # shutil.copyfile(data['image'], img_file)
        # shutil.copyfile(data['id'], id_file)
        # shutil.copyfile(data['label'], label_file)

        # print data['id']
        print k, counter, img_file
        counter = counter + 1

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
import random
from collections import OrderedDict
#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("raw_datasets_to_yolo")


seed = node.setupParameter("seed", -1)
if seed >= 0:
    random.seed(seed)
else:
    random.seed()


prefix = node.setupParameter("prefix", "yolo_")

#⬢⬢⬢⬢⬢➤ dataset
dataset_folder = node.setupParameter("dataset_folder", "")
dataset = RawDataset(dataset_folder)


#⬢⬢⬢⬢⬢➤ schema
schema = OrderedDict({
    "train": 0.8,
    "test": 0.2,
    "val": 0
})
residual_slot = "train"

data = dataset.data_list
random.shuffle(data)

size = len(data)

schema_sizes = {}
counter = 0
for name, perc in schema.iteritems():
    schema_sizes[name] = int(perc * size)
    counter = counter + int(perc * size)

diff = size - counter
schema_sizes[residual_slot] = schema_sizes[residual_slot] + diff


index = 0
for name, size in schema_sizes.iteritems():
    if size > 0:
        i1 = index
        i2 = i1 + size
        sub_data = data[i1:i2]

        rel_file = os.path.join(dataset_folder, prefix + name + ".txt")
        f = open(rel_file, 'w')
        for sd in sub_data:
            f.write(sd['image'] + "\n")
        f.close()

        index = i2

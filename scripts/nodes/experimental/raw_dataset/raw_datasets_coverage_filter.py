#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.datasets.datasetutils import TrainingScene, TrainingClassesMap, TrainingClass, TrainingFrame, TrainingDataset, RawDatasetBuilder, RawDataset
from roars.rosutils.rosnode import RosNode
from roars.datasets.datasetutils import JSONHelper
from roars.gui.pyqtutils import PyQtWindow, PyQtImageConverter, PyQtWidget
from roars.gui.widgets.WBaseWidget import WBaseWidget
from roars.gui.widgets.WSceneFrameVisualizer import WSceneFrameVisualizer
from roars.vision.augmentereality import VirtualObject
import roars.geometry.transformations as transformations
import PyQt4.QtGui as QtGui
import PyKDL
import sys
import cv2
import numpy as np
import functools
import random
import sys
import glob
import os
import collections
import math


def asSpherical(p):
    r = np.ravel(p)
    x = r[0]
    y = r[1]
    z = r[2]
    # takes list xyz (single coord)
    r = math.sqrt(x * x + y * y + z * z)
    theta = math.acos(z / r) * 180 / math.pi  # to degrees
    phi = math.atan2(y, x) * 180 / math.pi
    return [r, theta, phi]


#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("roars_dataset_export")


roars_datasets_path = node.setupParameter("roars_datasets_path", '')
raw_dataset_path = node.setupParameter("raw_dataset_path", "")
output_folder = node.setupParameter("output_folder", "/tmp")

files = glob.glob(os.path.join(roars_datasets_path, "*.txt"))
scenes = {}
for f in files:
    scene = TrainingScene.loadFromFile(f)
    if scene != None and scene.isValid():
        name = scene.getScenePath()
        scenes[name] = scene

for name, scene in scenes.iteritems():
    print name


raw_dataset = RawDataset(raw_dataset_path)


matched_frames = {}

for id_file, data in raw_dataset.data_map.iteritems():
    id = open(data['id'], 'r').readline()

    scene_name = id.split("!")[0]
    image_name = id.split("!")[1]
    number = int(image_name.split(".")[0].split("_")[1])

    if scene_name in scenes:
        scene = scenes[scene_name]
        pose = scene.getCameraPose(number)
        frame = scene.getFrameByIndex(number)
        matched_frames[id] = frame
        print id, matched_frames[id]


coverage_map = {}

filtered_ids = {}
for id, frame in matched_frames.iteritems():
    print "#########"
    print id, frame.getCameraPose()

    for inst in frame.getInstances():
        i_frame = PyKDL.Frame()
        i_frame.M = inst.M
        i_frame.p = inst.p

        i_frame = i_frame.Inverse()
        i_frame = i_frame * frame.getCameraPose()

        label = inst.label
        if label not in coverage_map:
            coverage_map[label] = []

        trans = transformations.KDLtoNumpyVector(i_frame)
        sph = asSpherical(trans)
        if label == 0 and sph[2] > 0 and sph[2] < 90:
            filtered_ids[id] = 1


output_ids = os.path.join(output_folder, "coverage_filtered_ids.txt")
f = open(output_ids, 'w')
for id, v in filtered_ids.iteritems():
    f.write(id + "\n")
f.close()

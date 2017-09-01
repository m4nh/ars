#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.datasets.datasetutils import TrainingScene, TrainingClassesMap, TrainingClass, TrainingFrame, TrainingDataset, RawDatasetBuilder
from roars.rosutils.rosnode import RosNode
from roars.datasets.datasetutils import JSONHelper
from roars.gui.pyqtutils import PyQtWindow, PyQtImageConverter, PyQtWidget
from roars.gui.widgets.WBaseWidget import WBaseWidget
from roars.gui.widgets.WSceneFrameVisualizer import WSceneFrameVisualizer
from roars.vision.augmentereality import VirtualObject
from PyQt4 import QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtGui import QFileDialog
import PyQt4.QtGui as QtGui
import PyKDL
import sys
import cv2
import numpy as np
import functools
import random
import sys

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("roars_dataset_export")

scene_manifest_file = node.setupParameter("scene_manifest_file", '')

output_folder = node.setupParameter(
    "output_folder", "")


WBaseWidget.DEFAULT_UI_WIDGETS_FOLDER = node.getFileInPackage(
    'roars', 'data/gui_forms/widgets'
)


scene = TrainingScene.loadFromFile(scene_manifest_file)

dataset = TrainingDataset([scene])
dataset_builder = RawDatasetBuilder(dataset, output_folder)

if dataset_builder.build():
    pass
else:
    print("Invalid outpu folder '{}'".format(output_folder))

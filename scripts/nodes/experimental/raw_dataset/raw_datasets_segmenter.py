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
from joblib import Parallel, delayed
import multiprocessing


class VoxelGrid(object):

    def __init__(self, center, side,  resolution):
        self.center = center
        self.side = side
        self.side_meter = side * resolution
        self.res = resolution
        self._voxels = np.zeros((side, side, side), dtype=np.int32)

    def getCoordinates(self, index):
        cx = index[0] * self.res + self.res * 0.5 + \
            self.center[0] - self.side_meter * 0.5
        cy = index[1] * self.res + self.res * 0.5 + \
            self.center[1] - self.side_meter * 0.5
        cz = index[2] * self.res + self.res * 0.5 + \
            self.center[2] - self.side_meter * 0.5
        return np.array([cx, cy, cz])

    def getIndex(self, point):
        ix = (point[0] - self.center[0] + self.side_meter *
              0.5 - self.res * 0.5) / self.res
        iy = (point[1] - self.center[1] + self.side_meter *
              0.5 - self.res * 0.5) / self.res
        iz = (point[2] - self.center[2] + self.side_meter *
              0.5 - self.res * 0.5) / self.res
        return np.array([ix, iy, iz])

    def carve(self, camera, camera_pose, image_mask):

        camera_inv = camera_pose.Inverse()
        out_points = np.zeros((1, 3, 1), dtype="float32")
        cam = np.array(camera.camera_matrix)
        dist = np.array(camera.distortion_coefficients)
        for ix in range(0, self.side):
            for iy in range(0, self.side):
                for iz in range(0, self.side):
                    p = self.getCoordinates(np.array([ix, iy, iz]))
                    point = PyKDL.Frame(PyKDL.Vector(p[0], p[1], p[2]))
                    object_to_camera = camera_inv * point
                    cRvec, cTvec = transformations.KDLToCv(object_to_camera)

                    point2d, _ = cv2.projectPoints(
                        out_points, cRvec, cTvec,
                        cam,
                        dist
                    )
                    point2d = point2d.reshape((2, 1))
                    i = int(point2d[1])
                    j = int(point2d[0])
                    if i < image_mask.shape[0] and j < image_mask.shape[1] and i >= 0 and j >= 0:
                        if image_mask[i, j] < 1:
                            self._voxels[
                                ix, iy, iz] = self._voxels[ix, iy, iz] + 1

        import scipy.io
        scipy.io.savemat("provola.mat", mdict={
                         'out': self._voxels}, oned_as='row')


def boxToRec(image, box):
    width = image.shape[1]
    height = image.shape[0]

    x = int(box[0] * width -
            box[2] * width * 0.5)
    y = int(box[1] * height -
            box[3] * height * 0.5)
    w = int(box[2] * width)
    h = int(box[3] * height)

    return ((x, y), (x + w, y + h))


def segment(image, rectangle, r1=0.3, r2=0.3, target_label=1, show_debug=False):
    min_y = min(rectangle[0][1], rectangle[1][1])
    min_x = min(rectangle[0][0], rectangle[1][0])

    roi = image[
        min(rectangle[0][1], rectangle[1][1]):max(rectangle[0][1], rectangle[1][1]),
        min(rectangle[0][0], rectangle[1][0]):max(rectangle[0][0], rectangle[1][0])
    ]
    markers = np.zeros((roi.shape[0], roi.shape[1], 1)).astype('int32')
    cx = int(markers.shape[1] / 2.0)
    cy = int(markers.shape[0] / 2.0)
    w = markers.shape[1]
    h = markers.shape[0]
    s1 = int(cx * r1)
    s2 = int(cx * r2)

    cv2.circle(markers, (cx, cy), s1, 1, -1)
    cv2.circle(markers, (0, 0), s2, 2, -1)
    cv2.circle(markers, (w, 0), s2, 2, -1)
    cv2.circle(markers, (w, h), s2, 2, -1)
    cv2.circle(markers, (0, h), s2, 2, -1)

    cv2.watershed(roi, markers)

    markers[markers != target_label] = 0
    markers[markers == target_label] = 255

    if show_debug:
        debug_image = np.zeros((roi.shape[0], roi.shape[1], 1))
        cv2.circle(debug_image, (cx, cy), s1, (255), -1)
        cv2.circle(debug_image, (0, 0), s2, (255), -1)
        cv2.circle(debug_image, (w, 0), s2, (255), -1)
        cv2.circle(debug_image, (w, h), s2, (255), -1)
        cv2.circle(debug_image, (0, h), s2, (255), -1)
        cv2.imshow("debug", debug_image)

    mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
    # mask[
    #     min(rectangle[0][1], rectangle[1][1]):max(rectangle[0][1], rectangle[1][1]),
    #     min(rectangle[0][0], rectangle[1][0]):max(rectangle[0][0], rectangle[1][0])
    # ] = markers
    cv2.circle(
        mask, (min_x + cx, min_y + cy), 25, (255), -1)
    return mask


def drawBox(image, box):
    width = image.shape[1]
    height = image.shape[0]

    w = int(box[3] * width)
    h = int(box[4] * height)
    x = int(box[1] * width)
    y = int(box[2] * height)
    cv2.rectangle(
        image,
        (int(x - w * 0.5), int(y - h * 0.5)),
        (int(x + w * 0.5), int(y + h * 0.5)),
        (255),
        2)


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
node = RosNode("roars_dataset_segmenter")


manifest_file = node.setupParameter("manifest_file", '')

#⬢⬢⬢⬢⬢➤ Create Scenes
scene = TrainingScene.loadFromFile(manifest_file)
target_label = 3

grid = VoxelGrid(center=np.array(
    [0.819, 0.125, -0.484]), side=64, resolution=0.005)


if scene.isValid():
    frames = scene.getAllFrames()

    for i in range(0, len(frames), 50):
        f = frames[i]
        image = cv2.imread(f.getImagePath())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = f.getInstancesGT()
        for b in boxes:
            if b[0] != target_label:
                continue

            drawBox(image, b)
            rec = boxToRec(image, b[1:5])
            mask = segment(image, rec, show_debug=True)
            img_mask = mask.copy()
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            image[mask > 0] = 0

            grid.carve(f.getCameraParams(), f.getCameraPose(), img_mask)

        cv2.imshow("img", image)
        c = cv2.waitKey(10)
        if c == 113 or c == 13:
            sys.exit(0)

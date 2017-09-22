#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.gui.pyqtutils import PyQtWidget, PyQtImageConverter
from roars.datasets.datasetutils import TrainingInstance, TrainingClass
import roars.geometry.lines as lines
from WBaseWidget import WBaseWidget
from WAxesEditor import WAxesEditor
from WAxesButtons import WAxesButtons
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4 import QtCore
import PyQt4.QtGui as QtGui
import math
import numpy as np
import PyKDL


class WInstanceCreator(WBaseWidget):

    def __init__(self, scene, name='CREATOR', changeCallback=None):
        super(WInstanceCreator, self).__init__(
            'ui_instance_creator'
        )
        self.name = name
        self.scene = scene
        self.boxes = []

        self.ui_button_create.clicked.connect(self.computeFrustumIntersection)

        #⬢⬢⬢⬢⬢➤ Change Callback
        self.changeCallback = changeCallback

        self.ui_check_enable.stateChanged.connect(self.enableEditing)
        self.ui_button_clear.clicked.connect(self.clear)

    def clear(self):
        self.reset()
        self.notifyEditing()

    def enableEditing(self, v):
        if self.changeCallback != None:
            self.notifyEditing()

    def notifyEditing(self):
        if self.ui_check_enable.isChecked():
            self.changeCallback((self.name, "ENABLE_EDITING"))
        else:
            self.changeCallback((self.name, "DISABLE_EDITING"))
            self.reset()

    def reset(self):
        self.boxes = []
        self.refresh()

    def addRawData(self, data):
        if data['type'] == 'BOX':
            self.boxes.append(data)
        self.refresh()

    def refresh(self):
        self.ui_label_counter.setText(str(len(self.boxes)))

    def computeFrustumIntersection(self):
        rays = []
        sizes = []
        target_camera_pose = None
        target_camera_matrix = None

        # compute RAYS inteserction
        for b in self.boxes:
            p1 = np.array(b['rect'][0])
            p2 = np.array(b['rect'][1])
            center = p1 + p2
            center = center * 0.5

            camera_pose = b['camera_pose']
            camera_inv = b['camera_matrix_inv']
            camera_matrix = b['camera_matrix']
            if target_camera_pose == None:
                target_camera_pose = camera_pose
                target_camera_matrix = camera_matrix
            ray3D = lines.compute3DRay(
                center, camera_inv, camera_pose)
            z = np.array([
                camera_pose.M.UnitZ().x(),
                camera_pose.M.UnitZ().y(),
                camera_pose.M.UnitZ().z()
            ])
            p = np.array([
                camera_pose.p.x(),
                camera_pose.p.y(),
                camera_pose.p.z()
            ])
            rays.append(ray3D)
            print ray3D
            sizes.append(np.linalg.norm(p1 - p2))

        point3D = lines.lineLineIntersection(rays)
        projected = self.computePointInCameraFrame(target_camera_pose, point3D)

        # compute depth of intersection
        d = projected[2]

        # compute average size of frustum far planes
        sx = 0
        for s in sizes:
            sx = sx + s
        sx = sx / float(len(sizes))
        ps = 0.8 * sx / target_camera_matrix[0][0] * d

        # build frame
        frame = PyKDL.Frame()
        frame.p = PyKDL.Vector(
            point3D[0], point3D[1], point3D[2] - ps * 0.5
        )
        self.changeCallback((self.name, "NEW_INSTANCE",
                             {
                                 'frame': frame, 'size': [ps, ps, ps]
                             }))
        self.reset()
        self.notifyEditing()

    def computePointInCameraFrame(self, camera_pose, point):
        pp = camera_pose.Inverse() * PyKDL.Vector(
            point[0],
            point[1],
            point[2]
        )
        return np.array([pp[0], pp[1], pp[2]])

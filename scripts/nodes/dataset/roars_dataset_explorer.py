#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.datasets.datasetutils import TrainingScene
from roars.rosutils.rosnode import RosNode
from roars.datasets.datasetutils import JSONHelper
from roars.gui.pyqtutils import PyQtWindow, PyQtImageConverter
from roars.vision.augmentereality import VirtualObject
from PyQt4 import QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyKDL
import sys
import cv2
import numpy as np
import functools

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("roars_dataset_explorer")

scene_manifest_file = node.setupParameter("scene_manifest_file", '')

#⬢⬢⬢⬢⬢➤ Create Scenes
scene = TrainingScene.loadFromFile(scene_manifest_file)


#⬢⬢⬢⬢⬢➤ Save Scene to file if it is valid
if scene.isValid():
    print("Scene ready!")
else:
    print("Scene is not valid!")
    sys.exit(0)


print(scene.robot_to_camera_pose)

frames = scene.getAllFrames()
print(len(frames))

current_frame = 0


class MainWindow(PyQtWindow):

    def __init__(self, uifile, scene):
        super(MainWindow, self).__init__(uifile)
        self.image.mousePressEvent = self.getPos

        print("Style", self.styleSheet())

        #⬢⬢⬢⬢⬢➤ Frame Management
        self.scene = scene
        self.frames = scene.getAllFrames()
        self.current_frame_index = -1
        self.current_frame = None
        self.current_image = np.zeros((50, 50))

        #⬢⬢⬢⬢⬢➤ Instances_management
        self.temporary_instances = []
        self.selected_instance = -1
        self.ui_spin_frame_roll.valueChanged.connect(self.frameValuesChanged)
        self.ui_spin_frame_z.valueChanged.connect(self.frameValuesChanged)

        ##
        self.ui_button_next_frame.clicked.connect(self.nextFrame)
        self.ui_button_back_frame.clicked.connect(self.backFrame)

        ##
        self.ui_button_load_raw_objects.clicked.connect(self.loadRawObjects)
        self.ui_button_load_classification.clicked.connect(
            self.loadClassificationCfg)

    def getSelectedInstance(self):
        try:
            return self.temporary_instances[self.selected_instance]
        except:
            return None

    def frameValuesChanged(self,  v):
        inst = self.getSelectedInstance()
        if inst != None:
            obj_name = self.sender().objectName()
            if 'roll' in obj_name:
                roll = v * np.pi / 180.0
                inst.setRPY(roll, None, None)

            if 'z' in obj_name:
                z = v
                inst.p.z(z)

            self.refresh()

    def loadClassificationCfg(self):
        fname = QFileDialog.getOpenFileName(
            self, 'Load Classification Configuration', '', "Roars Classification Configurations (*.clc)")

        cfg = JSONHelper.loadFromFile(fname)
        self.list_classes.clear()
        for cl in cfg["classes"]:
            self.list_classes.insertItem(self.list_classes.count(), cl)
        self.label_classification_configuration_name.setText(cfg["name"])

    def loadRawObjects(self):
        fname = QFileDialog.getOpenFileName(
            self,
            'Load Raw Objects List',
            '',
            "Arp File (*.arp)"
        )
        obj_data = JSONHelper.loadFromFile(fname)

        self.temporary_instances = obj_data["objects_instances"]
        self.refreshInstacesList()

    def refreshInstacesList(self):

        list_model = QStandardItemModel(self.ui_listm_instances)

        for i in range(0, len(self.temporary_instances)):
            inst = self.temporary_instances[i]
            item = QStandardItem()
            item.setText("Instance_{}".format(i))
            item.setCheckable(True)
            list_model.appendRow(item)

        self.ui_listm_instances.setModel(list_model)
        self.ui_listm_instances.selectionModel().currentChanged.connect(
            self.listInstancesSelectionChange)

    def listInstancesSelectionChange(self, current, previous):
        self.selectInstance(current.row())

    def selectInstance(self, index):
        self.selected_instance = index
        if self.selected_instance >= 0:
            inst = self.temporary_instances[index]
            inst_rpy = inst.getRPY()

            #⬢⬢⬢⬢⬢➤ Update gui fields
            self.ui_spin_frame_roll.setValue(inst_rpy[0] * 180.0 / np.pi)
            self.ui_spin_frame_z.setValue(inst.p.z())

            print(inst_rpy)
        self.refresh()

    def nextFrame(self):

        step = self.ui_spin_frame_step.value()
        self.current_frame_index += step
        self.current_frame_index = self.current_frame_index % len(self.frames)
        self.updateCurrentFrame()

    def backFrame(self):
        step = self.ui_spin_frame_step.value()
        self.current_frame_index -= step
        self.current_frame_index = self.current_frame_index % len(self.frames)
        self.updateCurrentFrame()

    def updateCurrentFrame(self):
        self.current_frame = self.frames[self.current_frame_index]
        self.current_image = cv2.imread(self.current_frame.getImagePath())
        self.refresh()

    def drawInstances(self, img):
        for i in range(0, len(self.temporary_instances)):
            inst = self.temporary_instances[i]
            vo = VirtualObject(frame=inst, size=inst.size, label=inst.label)
            thick = 1 if self.selected_instance != i else 5
            vo.draw(
                img,
                camera_frame=self.current_frame.getCameraPose(),
                camera=self.scene.camera_params,
                thickness=thick
            )

    def refresh(self):

        display_image = self.current_image.copy()
        self.drawInstances(display_image)

        pix = PyQtImageConverter.cvToQPixmap(display_image)
        pix = pix.scaled(self.image.size(), QtCore.Qt.KeepAspectRatio)
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setPixmap(pix)

        self.label_current_frame.setText(
            "Current Frame: {}".format(self.current_frame_index + 1))

    def getPos(self, event):
        x = event.pos().x()
        y = event.pos().y()
        print(x, y)


window = MainWindow(
    uifile='/home/daniele/work/ros/roars_ws/src/roars/data/gui_forms/arp_gui.ui',
    scene=scene
)
window.run()


while node.isActive():

    frame = frames[current_frame]

    img = cv2.imread(frame.image_path)

    PyQtImageConverter.cvToQt(img)
    cv2.imshow("prova", img)
    c = cv2.waitKey(1)
    print(c)

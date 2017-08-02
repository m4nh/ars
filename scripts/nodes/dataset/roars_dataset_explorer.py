#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.datasets.datasetutils import TrainingScene, TrainingClassesMap, TrainingClass
from roars.rosutils.rosnode import RosNode
from roars.datasets.datasetutils import JSONHelper
from roars.gui.pyqtutils import PyQtWindow, PyQtImageConverter
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

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("roars_dataset_explorer")

scene_manifest_file = node.setupParameter("scene_manifest_file", '')


class MainWindow(PyQtWindow):

    def __init__(self, uifile):
        super(MainWindow, self).__init__(uifile)

        #⬢⬢⬢⬢⬢➤ Scene Management
        self.scene_filename = ''
        self.scene = None
        self.frames = None

        #⬢⬢⬢⬢⬢➤ Frame Management
        self.current_frame_index = -1
        self.current_frame = None
        self.current_image = np.zeros((50, 50))
        self.ui_button_next_frame.clicked.connect(self.nextFrame)
        self.ui_button_back_frame.clicked.connect(self.backFrame)

        #⬢⬢⬢⬢⬢➤ Instances_management
        self.ui_button_load_raw_objects.clicked.connect(
            self.loadRawObjects
        )
        self.ui_combo_frame_classes.currentIndexChanged.connect(
            self.comboBoxChanged
        )
        self.temporary_instances = []
        self.selected_instance = -1
        self.frame_coordinates_attributes = {
            "cx": 1, "cy": 1, "cz": 1, "roll": np.pi / 180.0, "pitch": np.pi / 180.0, "yaw": np.pi / 180.0}
        for attr, _ in self.frame_coordinates_attributes.iteritems():
            name = "ui_spin_frame_{}".format(attr)
            getattr(self, name).valueChanged.connect(self.frameValuesChanged)

        #⬢⬢⬢⬢⬢➤ Classes Management
        self.temporary_class_map = None
        self.ui_button_load_classification.clicked.connect(
            self.loadClassificationCfg)
        self.ui_list_classes.currentIndexChanged.connect(self.comboBoxChanged)

        #⬢⬢⬢⬢⬢➤ Storage Management
        self.ui_button_save.clicked.connect(self.save)

    def initScene(self, scene_filename=''):

        self.scene_filename = scene_filename

        #⬢⬢⬢⬢⬢➤ Create Scenes
        scene = TrainingScene.loadFromFile(scene_manifest_file)

        if scene == None:
            self.showDialog("Scene file is not valid!")
            sys.exit(0)

        if not scene.isValid():
            self.showDialog("Scene file is corrupted!")
            sys.exit(0)

        #⬢⬢⬢⬢⬢➤ Init
        self.scene = scene
        self.frames = scene.getAllFrames()

        self.nextFrame()

        #⬢⬢⬢⬢⬢➤ Check Ready Classes/INstances
        if len(self.scene.classes) > 0:
            self.temporary_instances = self.scene.getAllInstances()
            self.refreshInstacesList()
            self.setClassMap(self.scene.generateClassesMap())

        self.refresh()

    def save(self):
        self.scene.setClasses(
            TrainingClass.generateClassListFromInstances(
                self.temporary_instances, classes_map=self.temporary_class_map)
        )
        if self.showPromptBool(title='Saving Scene', message='Are you sure?'):
            self.scene.save(self.scene_filename)

    def comboBoxChanged(self, index):
        combo = self.sender()
        if "classes" in combo.objectName():
            color = TrainingClass.getColorByLabel(index - 1, output_type="HEX")
            combo.setStyleSheet(
                "QComboBox::drop-down {background: " + color + ";}")

            inst = self.getSelectedInstance()
            if inst:
                inst.label = index - 1

            self.refresh()

    def getSelectedInstance(self):
        try:
            return self.temporary_instances[self.selected_instance]
        except:
            return None

    def frameValuesChanged(self,  v):
        inst = self.getSelectedInstance()
        if inst != None:
            obj_name = self.sender().objectName()
            for attr, conv in self.frame_coordinates_attributes.iteritems():
                if attr in obj_name:
                    inst.setFrameProperty(attr, v * conv)

            self.refresh()

    def setClassMap(self, class_map):
        self.temporary_class_map = class_map
        self.updateListWithClassesMap(
            self.ui_list_classes,
            self.temporary_class_map
        )
        self.updateListWithClassesMap(
            self.ui_combo_frame_classes,
            self.temporary_class_map
        )

    def loadClassificationCfg(self):
        fname = QFileDialog.getOpenFileName(
            self,
            'Load Classification Configuration',
            '',
            "Roars Classification Configurations (*.clc)"
        )

        cfg = JSONHelper.loadFromFile(fname)
        self.setClassMap(TrainingClassesMap(cfg["classes"]))

    def updateListWithClassesMap(self, ui_list, class_map):
        ui_list.clear()
        model = ui_list.model()
        for k, v in class_map.map().iteritems():
            print("@", k, v)
            item = QtGui.QStandardItem(v)
            color = TrainingClass.getColorByLabel(k)
            item.setForeground(QtGui.QColor(color[2], color[1], color[0]))
            font = item.font()
            font.setPointSize(10)
            item.setFont(font)
            model.appendRow(item)
            #ui_list.insertItem(ui_list.count(), item)

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
            item.setCheckable(False)
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
            for attr, conv in self.frame_coordinates_attributes.iteritems():
                name = "ui_spin_frame_{}".format(attr)
                getattr(self, name).setValue(
                    inst.getFrameProperty(attr) / conv)

            self.ui_combo_frame_classes.setCurrentIndex(inst.label + 1)

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
            thick = 1 if self.selected_instance != i else 4

            color = TrainingClass.getColorByLabel(
                inst.label, output_type="RGB")

            vo.draw(
                img,
                camera_frame=self.current_frame.getCameraPose(),
                camera=self.scene.camera_params,
                thickness=thick,
                color=color
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

    def temp(self, v):
        print(self.ui_list_classes.itemText(v))


window = MainWindow(
    uifile='/home/daniele/work/ros/roars_ws/src/roars/data/gui_forms/arp_gui.ui'
)
window.initScene(scene_filename=scene_manifest_file)
window.run()


while node.isActive():

    frame = frames[current_frame]

    img = cv2.imread(frame.image_path)

    PyQtImageConverter.cvToQt(img)
    cv2.imshow("prova", img)
    c = cv2.waitKey(1)
    print(c)

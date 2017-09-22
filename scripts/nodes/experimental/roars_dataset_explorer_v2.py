#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.datasets.datasetutils import TrainingScene, TrainingClassesMap, TrainingClass, TrainingInstance
from roars.rosutils.rosnode import RosNode
from roars.datasets.datasetutils import JSONHelper
from roars.gui.pyqtutils import PyQtWindow, PyQtImageConverter, PyQtWidget
from roars.gui.widgets.WBaseWidget import WBaseWidget
from roars.gui.widgets.WInstanceEditor import WInstanceEditor
from roars.gui.widgets.WSceneFrameVisualizer import WSceneFrameVisualizer
from roars.gui.widgets.WInstanceCreator import WInstanceCreator
from roars.gui.widgets.WAxesEditor import WAxesEditor
from roars.vision.augmentereality import VirtualObject
import roars.vision.cvutils as cvutils
import roars.geometry.lines as lines
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
import sys
import math
import re


#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("roars_dataset_explorer_v2")
window_size = node.setupParameter("window_size", "1400;900", array_type=int)

scene_manifest_file = node.setupParameter("scene_manifest_file", '')

WBaseWidget.DEFAULT_UI_WIDGETS_FOLDER = node.getFileInPackage(
    'roars', 'data/gui_forms/widgets'
)
PyQtWindow.DEFAULT_IMAGES_PATH = node.getFileInPackage(
    'roars', 'data/gui_forms/images'
)


class MainWindow(PyQtWindow):

    def __init__(self, uifile):
        super(MainWindow, self).__init__(uifile)

        # self.showMaximized()
        self.setFixedSize(window_size[0], window_size[1])
        #⬢⬢⬢⬢⬢➤ Scene Management
        self.scene_filename = ''
        self.scene = None
        self.frames = None

        #⬢⬢⬢⬢⬢➤ Frame Visualizers Management
        self.ui_scene_visualizers_list = [
            WSceneFrameVisualizer(),
            WSceneFrameVisualizer(),
            WSceneFrameVisualizer(),
            WSceneFrameVisualizer(),
            WSceneFrameVisualizer()
        ]
        self.ui_view_single_container.addWidget(
            self.ui_scene_visualizers_list[0]
        )
        for fv in self.ui_scene_visualizers_list:
            fv.mousePressCallback = self.frameClickedPointCallback

        self.scene_visualizers_points = []
        self.scene_visualizers_boxes = []

        self.ui_view_container_1.addWidget(self.ui_scene_visualizers_list[1])
        self.ui_view_container_2.addWidget(self.ui_scene_visualizers_list[2])
        self.ui_view_container_3.addWidget(self.ui_scene_visualizers_list[3])
        self.ui_view_container_4.addWidget(self.ui_scene_visualizers_list[4])
        self.ui_main_tab.currentChanged.connect(self.refreshVisualizers)
        self.ui_button_randomize_views.clicked.connect(
            self.randomizeVisualizers)

        # Instance Editor
        self.ui_instance_editor = WInstanceEditor(changeCallback=self.refresh)
        self.ui_test_layout.addWidget(self.ui_instance_editor)

        # Instance Creator
        self.ui_instance_creator = WInstanceCreator(
            scene=self.scene,
            changeCallback=self.creatorCallback
        )
        self.ui_test_layout.addWidget(self.ui_instance_creator)
        for sv in self.ui_scene_visualizers_list:
            sv.addDrawerCallback(self.ui_instance_creator.addRawData)

        #⬢⬢⬢⬢⬢➤ Instances_management
        self.ui_button_load_raw_objects.clicked.connect(
            self.loadRawObjects
        )

        #⬢⬢⬢⬢⬢➤ Classes Management
        self.temporary_class_map = None
        self.ui_button_load_classification.clicked.connect(
            self.loadClassificationCfg)
        self.ui_list_classes.currentIndexChanged.connect(self.comboBoxChanged)

        #⬢⬢⬢⬢⬢➤ Storage Management
        self.ui_button_save.clicked.connect(self.save)

    def testChange(self, data):
        print("CHANGE", data)

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

        #⬢⬢⬢⬢⬢➤ Check Ready Classes/INstances
        if len(self.scene.classes) > 0:
            self.updateClassLists(scene.classes)
            self.setInstances(scene.getAllInstances())
            pass

        self.initSceneForVisualizers(scene)
        self.randomizeVisualizers()

        QtCore.QTimer.singleShot(200, self.refreshVisualizers)

    def resizeEvent(self, event):
        print("resize")
        # self.refreshVisualizers()

    def initSceneForVisualizers(self, scene):
        for ui in self.ui_scene_visualizers_list:
            ui.initScene(self.scene)

    def randomizeVisualizers(self):
        for ui in self.ui_scene_visualizers_list:
            ui.randomFrame()

    def refreshVisualizers(self):
        for ui in self.ui_scene_visualizers_list:
            ui.refresh()

    def creatorCallback(self, data):
        if data[1] == "ENABLE_EDITING":
            for ui in self.ui_scene_visualizers_list:
                ui.enableInteraction()
        elif data[1] == "DISABLE_EDITING":
            for ui in self.ui_scene_visualizers_list:
                ui.enableInteraction(False)
        elif data[1] == "NEW_INSTANCE":
            print("NEW ISNTANCE", data[2])
            self.createNewInstance(data[2])

    def save(self):
        if self.showPromptBool(title='Saving Scene', message='Are you sure?'):
            self.scene.setInstances(self.temporary_instances)
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

    def updateClassLists(self, class_map):
        self.ui_instance_editor.setClassMap(class_map)

    def setClassMap(self, class_map):
        self.scene.setClasses(class_map.getClasses())
        self.updateClassLists(self.scene.classes)

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
        labels = sorted(class_map.keys())

        for k in labels:
            v = class_map[k]
            print("@", k, v.name)
            item = QtGui.QStandardItem(v.name)
            color = TrainingClass.getColorByLabel(k)
            item.setForeground(QtGui.QColor(color[2], color[1], color[0]))
            font = item.font()
            font.setPointSize(10)
            item.setFont(font)
            model.appendRow(item)
            # ui_list.insertItem(ui_list.count(), item)

    def loadRawObjects(self):
        if len(self.scene.classes) <= 0:
            self.showDialog(text='No Classes Configuration Loaded!')
            return

        fname = QFileDialog.getOpenFileName(
            self,
            'Load Raw Objects List',
            '',
            "Arp File (*.arp)"
        )
        obj_data = JSONHelper.loadFromFile(fname)

        self.scene.getTrainingClass(-1).addInstances(
            obj_data["objects_instances"]
        )
        self.setInstances(obj_data["objects_instances"])

    def setInstances(self, instances):
        self.temporary_instances = instances
        self.refreshInstacesList(self.scene.getAllInstances())

    def refreshInstances(self):
        self.refreshInstacesList(self.scene.getAllInstances())

    def refreshInstacesList(self, instances):
        list_model = QStandardItemModel(self.ui_listm_instances)
        list_model.clear()
        for i in range(0, len(instances)):
            inst = instances[i]
            item = QStandardItem()
            item.setText("Instance_{}".format(i))
            item.setCheckable(False)
            list_model.appendRow(item)

        self.ui_listm_instances.setModel(list_model)
        self.ui_listm_instances.selectionModel().currentChanged.connect(
            self.listInstancesSelectionChange)

    def listInstancesSelectionChange(self, current, previous):
        self.selectInstance(current.row())

    def updateInstanceValues(self, instance):
        #⬢⬢⬢⬢⬢➤ Update gui fields
        for attr, conv in self.frame_coordinates_attributes.iteritems():
            name = "ui_spin_frame_{}".format(attr)
            getattr(self, name).setValue(
                instance.getFrameProperty(attr) / conv)

    def selectInstance(self, index):
        self.selected_instance = index
        if self.selected_instance >= 0:
            inst = self.temporary_instances[index]
            self.ui_instance_editor.setInstance(inst)
            for ui in self.ui_scene_visualizers_list:
                ui.setSelectedInstance(inst)

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

    def refresh(self):
        self.refreshVisualizers()

    def createNewInstance(self, data):
        training_class = self.scene.getTrainingClass(-1, force_creation=True)
        inst = TrainingInstance(frame=data["frame"], size=data["size"])
        training_class.instances.append(inst)

        self.setInstances(self.scene.getAllInstances())
        self.refresh()

    def frameClickedPointCallback(self, frame, data, action):
        if action == 'ADD':
            ray3D = lines.compute3DRay(
                data, self.scene.camera_params.camera_matrix_inv, frame.getCameraPose())

            z = np.array([
                frame.getCameraPose().M.UnitZ().x(),
                frame.getCameraPose().M.UnitZ().y(),
                frame.getCameraPose().M.UnitZ().z()
            ])
            p = np.array([
                frame.getCameraPose().p.x(),
                frame.getCameraPose().p.y(),
                frame.getCameraPose().p.z()
            ])
            print ray3D
            print "###"
            print p, z
            self.scene_visualizers_points.append(ray3D)

            rays_size = len(self.scene_visualizers_points)
            print("Rays", rays_size)
            if rays_size >= 4:
                x = lines.lineLineIntersection(self.scene_visualizers_points)
                print x

        if action == 'ADD_BOX':
            center = np.array(data[0]) + np.array(data[1])
            center = center * 0.5

            ray3D = lines.compute3DRay(
                center, self.scene.camera_params.camera_matrix_inv, frame.getCameraPose())
            z = np.array([
                frame.getCameraPose().M.UnitZ().x(),
                frame.getCameraPose().M.UnitZ().y(),
                frame.getCameraPose().M.UnitZ().z()
            ])
            p = np.array([
                frame.getCameraPose().p.x(),
                frame.getCameraPose().p.y(),
                frame.getCameraPose().p.z()
            ])
            print ray3D
            print "###"
            print p, z
            self.scene_visualizers_points.append(ray3D)
            self.scene_visualizers_boxes.append(data)
            rays_size = len(self.scene_visualizers_points)
            print("Rays", rays_size)
            if rays_size >= 2:
                x = lines.lineLineIntersection(self.scene_visualizers_points)
                print x


gui_file = node.getFileInPackage(
    'roars', 'data/gui_forms/roars_labeler_window.ui'


)
window = MainWindow(
    uifile=gui_file
)
window.initScene(scene_filename=scene_manifest_file)
window.run()

#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.datasets.datasetutils import PixDatasetBuilder, RawDatasetBuilder, TrainingScene, TrainingClassesMap, TrainingClass, TrainingInstance, TrainingDataset
from roars.rosutils.rosnode import RosNode
from roars.datasets.datasetutils import JSONHelper
from roars.gui.pyqtutils import PyQtWindow, PyQtImageConverter, PyQtWidget
from roars.gui.widgets.WBaseWidget import WBaseWidget
from roars.gui.widgets.WInstanceEditor import WInstanceEditor
from roars.gui.widgets.WSceneFrameVisualizer import WSceneFrameVisualizer
from roars.gui.widgets.WInstanceCreator import WInstanceCreator
from roars.gui.widgets.WInstancesList import WInstancesList
from roars.gui.widgets.WAxesEditor import WAxesEditor
from PyQt4 import QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtGui import QFileDialog
import PyQt4.QtGui as QtGui
import sys
import math
import os


#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("roars_scene_explorer")
window_size = node.setupParameter("window_size", "1400;900", array_type=int)
scene_manifest_file = node.setupParameter("manifest", '')

WBaseWidget.DEFAULT_UI_WIDGETS_FOLDER = node.getFileInPackage(
    'roars', 'data/gui_forms/widgets'
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

        self.scene_visualizers_points = []
        self.scene_visualizers_boxes = []

        self.ui_view_container_1.addWidget(self.ui_scene_visualizers_list[1])
        self.ui_view_container_2.addWidget(self.ui_scene_visualizers_list[2])
        self.ui_view_container_3.addWidget(self.ui_scene_visualizers_list[3])
        self.ui_view_container_4.addWidget(self.ui_scene_visualizers_list[4])
        self.ui_main_tab.currentChanged.connect(self.refreshVisualizers)
        self.ui_button_randomize_views.clicked.connect(
            self.randomizeVisualizers)

        # Instance List
        self.ui_instances_list = WInstancesList(
            changeCallback=self.instancesListChange, newCallback=self.createNewInstance)
        self.ui_test_layout.addWidget(self.ui_instances_list)

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

        #⬢⬢⬢⬢⬢➤ Classes Management
        self.temporary_class_map = None

        #⬢⬢⬢⬢⬢➤ Storage Management
        self.ui_button_save.clicked.connect(self.save)
        self.ui_button_generate.clicked.connect(self.generate)

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

        self.initSceneForVisualizers(scene)
        self.randomizeVisualizers()

        QtCore.QTimer.singleShot(200, self.refreshVisualizers)

    def resizeEvent(self, event):
        pass

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
            self.scene.setInstances(self.ui_instances_list.getInstances())
            self.scene.save(self.scene_filename)

    def generate(self):
        dataset_type = 'RAW'  # TODO: configuration or option
        dname = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        basename = os.path.basename(scene_manifest_file)
        basename = os.path.splitext(basename)[0]

        outpath = os.path.join(dname, basename + "_raw")
        if os.path.exists(outpath):
            self.showDialog('Folder {} is not void!'.format(outpath))
            return

        dataset = TrainingDataset([self.scene])
        print(dataset_type)
        if dataset_type == 'RAW':
            dataset_builder = RawDatasetBuilder(dataset, outpath)

            if dataset_builder.build():
                self.showDialog('Dataset built! Folder: {}'.format(outpath))
            else:
                self.showDialog('Invalid output folder: {}'.format(outpath))

        if dataset_type == 'PIX':

            dataset_builder = PixDatasetBuilder(
                dataset, outpath, jumps=1, val_percentage=0.0, test_percentage=0.0)
            if dataset_builder.build():
                print("PINO")

        if dataset_type == 'MASK':
            from roars.datasets.generators.maskgenerator import MaskDatasetBuilder
            dataset_builder = MaskDatasetBuilder(dataset, outpath, jumps=1, boxed_instances=True)
            if dataset_builder.build():
                self.showDialog('Dataset built! Folder: {}'.format(outpath))
            else:
                self.showDialog('Invalid output folder: {}'.format(outpath))

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
        self.ui_instances_list.setInstances(instances)

    def instancesListChange(self, inst):
        for ui in self.ui_scene_visualizers_list:
            ui.setSelectedInstance(inst)
            ui.refresh()
        self.ui_instance_editor.setSelectedInstance(inst)

    def refresh(self):
        self.refreshVisualizers()

    def createNewInstance(self, data):
        training_class = self.scene.getTrainingClass(-1, force_creation=True)
        inst = TrainingInstance(frame=data["frame"], size=data["size"])
        training_class.instances.append(inst)

        self.setInstances(self.scene.getAllInstances())
        self.refresh()


gui_file = node.getFileInPackage(
    'roars', 'data/gui_forms/roars_labeler_window.ui'
)
window = MainWindow(
    uifile=gui_file
)
window.initScene(scene_filename=scene_manifest_file)
window.run()

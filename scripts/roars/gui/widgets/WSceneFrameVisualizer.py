from roars.vision.augmentereality import VirtualObject
from roars.datasets.datasetutils import TrainingClass
from roars.gui.pyqtutils import PyQtWidget, PyQtImageConverter
from WBaseWidget import WBaseWidget
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4 import QtCore
import numpy as np
import cv2
import random


class WSceneFrameVisualizer(WBaseWidget):

    def __init__(self):
        super(WSceneFrameVisualizer, self).__init__(
            'ui_scene_frame_visualizer'
        )

        self.scene = None
        self.frames = []
        self.selected_instance = None

        self.current_frame_index = -1
        self.current_frame = None
        self.current_image = np.zeros((50, 50))
        self.ui_button_next_frame.clicked.connect(self.nextFrame)
        self.ui_button_prev_frame.clicked.connect(self.prevFrame)

    def setSelectedInstance(self, instance):
        self.selected_instance = instance

    def initScene(self, scene):
        self.scene = scene
        self.frames = scene.getAllFrames()
        self.current_frame_index = -1
        self.current_frame = None

    def randomFrame(self):
        if self.scene:
            self.current_frame_index = random.randint(0, len(self.frames) - 1)
            self.updateCurrentFrame()

    def nextFrame(self):
        if self.scene:
            step = self.ui_spin_frame_step.value()
            self.current_frame_index += step
            self.current_frame_index = self.current_frame_index % len(
                self.frames)
            self.updateCurrentFrame()

    def prevFrame(self):
        if self.scene:
            step = self.ui_spin_frame_step.value()
            self.current_frame_index -= step
            self.current_frame_index = self.current_frame_index % len(
                self.frames)
            self.updateCurrentFrame()

    def updateCurrentFrame(self):
        if self.scene:
            self.current_frame = self.frames[self.current_frame_index]
            self.current_image = cv2.imread(self.current_frame.getImagePath())
            self.refresh()

    def drawInstances(self, img):
        if self.scene:
            instances = self.scene.getAllInstances()
            for i in range(0, len(instances)):
                inst = instances[i]
                vo = VirtualObject(frame=inst, size=inst.size,
                                   label=inst.label)
                thick = 1 if self.selected_instance != inst else 4

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
        if self.scene:
            display_image = self.current_image.copy()
            self.drawInstances(display_image)

            pix = PyQtImageConverter.cvToQPixmap(display_image)
            pix = pix.scaled(self.ui_image.size(), QtCore.Qt.KeepAspectRatio)
            self.ui_image.setAlignment(QtCore.Qt.AlignCenter)
            self.ui_image.setPixmap(pix)

            self.ui_label_current_frame.setText(
                "F[{}]".format(self.current_frame_index + 1))

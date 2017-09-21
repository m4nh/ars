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
import PyKDL


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
        self.current_image_gizmo = np.zeros((50, 50))
        self.ui_button_next_frame.clicked.connect(self.nextFrame)
        self.ui_button_prev_frame.clicked.connect(self.prevFrame)

        self.ui_image.mousePressEvent = self.mousePressEvent
        self.ui_image.mouseMoveEvent = self.mouseMoveEvent
        self.mousePressCallback = None
        self.enable_click_feedback = True
        self.dragging = False

        self.ui_spin_frame_step.setValue(5)
        # DEBUG
        self.clicked_points = []

    def clearClickedPoints(self):
        self.clicked_points = []
        self.updateCurrentFrame()

    def mouseMoveEvent(self, evt):
        pos = self.imageCoordinatesFromFrameCoordinates(evt.x(), evt.y())
        if len(self.clicked_points) == 1:
            self.current_image_gizmo = self.current_image.copy()
            cv2.rectangle(
                self.current_image_gizmo, self.clicked_points[0], pos, (249, 202, 144), 2)
            cv2.circle(self.current_image_gizmo,
                       self.clicked_points[0], 5, (243, 150, 33), -1)
            cv2.circle(self.current_image_gizmo, pos, 5, (243, 150, 33), -1)
            self.refresh()

    def mousePressEvent(self, evt):
        # print 'click', evt.button(), evt.x(), evt.y()
        # pos = self.imageCoordinatesFromFrameCoordinates(evt.x(), evt.y())
        # if self.enable_click_feedback:
        #     if evt.button() == 1:
        #         cv2.circle(self.current_image, pos, 5, (0, 255, 255), -1)
        #         self.refresh()
        #         if self.mousePressCallback != None:
        #             self.mousePressCallback(
        #                 self.current_frame, pos, action='ADD')
        print 'click', evt.button(), evt.x(), evt.y()
        pos = self.imageCoordinatesFromFrameCoordinates(evt.x(), evt.y())
        if self.enable_click_feedback:
            if evt.button() == 1:
                self.current_image_gizmo = self.current_image.copy()
                self.clicked_points.append(pos)

                if len(self.clicked_points) == 2:

                    cv2.rectangle(
                        self.current_image_gizmo, self.clicked_points[0], pos, (243, 150, 33), 2)
                    cv2.circle(self.current_image_gizmo,
                               self.clicked_points[0], 4, (243, 150, 33), -1)
                    cv2.circle(self.current_image_gizmo,
                               pos, 4, (243, 150, 33), -1)
                    self.refresh()
                    if self.mousePressCallback != None:
                        self.mousePressCallback(
                            self.current_frame, (self.clicked_points[0], self.clicked_points[1]), action='ADD_BOX')

                #cv2.circle(self.current_image_gizmo, pos, 5, (0, 255, 255), -1)
                self.refresh()

    def imageCoordinatesFromFrameCoordinates(self, x, y, tp=int):
        img_size = self.current_image.shape
        img_w = float(img_size[1])
        img_h = float(img_size[0])

        frame_w = float(self.ui_image.size().width())
        frame_h = float(self.ui_image.size().height())

        h_ratio = frame_h / img_h
        img_reduced_w = img_w * h_ratio
        img_reduced_h = frame_h
        w_padding = (frame_w - img_reduced_w) * 0.5

        inner_x = (x - w_padding) / img_reduced_w
        inner_y = (y) / img_reduced_h
        img_x = inner_x * img_w
        img_y = inner_y * img_h

        print img_size, (frame_w, frame_h), h_ratio, img_reduced_w, w_padding
        print img_x, img_y
        return (tp(img_x), tp(img_y))

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
            self.current_image_gizmo = self.current_image.copy()
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
            display_image = cv2.cvtColor(
                self .current_image_gizmo, cv2.COLOR_BGR2RGB)

            self.drawInstances(display_image)

            pix = PyQtImageConverter.cvToQPixmap(display_image)

            pix = pix.scaled(self.ui_image.size().width()-5,self.ui_image.size().height()-1, QtCore.Qt.KeepAspectRatio)

            self.ui_image.setAlignment(QtCore.Qt.AlignCenter)
            self.ui_image.setPixmap(pix)
            
            self.ui_label_current_frame.setText(
                "F[{}]".format(self.current_frame_index + 1))

    def debugComputePointInCameraFrame(self, point):

        pp = self.current_frame.getCameraPose().Inverse() * PyKDL.Vector(
            point[0],
            point[1],
            point[2]
        )
        return np.array([pp[0], pp[1], pp[2]])

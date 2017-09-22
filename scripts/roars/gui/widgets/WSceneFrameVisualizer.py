from roars.vision.augmentereality import VirtualObject
from roars.datasets.datasetutils import TrainingClass
from roars.gui.pyqtutils import PyQtWidget, PyQtImageConverter
import roars.vision.colors as colors
from WBaseWidget import WBaseWidget
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4 import QtCore
import numpy as np
import cv2
import random
import PyKDL


class MouseDrawer(object):

    def __init__(self, widget, namespace='generic', callback=None):
        self.parent = widget
        self.namespace = namespace
        self.callback = callback

    def relativeCoordinates(self, evt):
        return self.parent.imageCoordinatesFromFrameCoordinates(
            evt.x(), evt.y())

    def getCurrentData(self):
        return (self.namespace, None)

    def reset(self):
        pass


class MouseBoxDrawer(MouseDrawer):
    DEFAULT_BOX_COLOR = (249, 202, 144)
    DEFAULT_BOX_EDGES_COLOR = (243, 150, 33)

    def __init__(self, widget, namespace='', callback=None):
        super(MouseBoxDrawer, self).__init__(widget, namespace, callback)
        self.p1 = None
        self.p2 = None

    def reset(self):
        self.p1 = self.p2 = None

    def manageClick(self, evt):
        if evt.button() == 1:
            pos = self.relativeCoordinates(evt)
            self.p1 = pos
            self.p2 = None
        if evt.button() == 2:
            self.p1 = self.p2 = None

    def manageMouseMove(self, evt):
        pos = self.relativeCoordinates(evt)
        if self.p1 != None:
            self.p2 = pos

    def manageMouseRelease(self, evt):
        if self.p1 != None and self.p2 != None:
            if self.callback != None:
                self.callback(self.getCurrentData())

    def drawGizmo(self, image):
        if self.p1 != None and self.p2 != None:
            cv2.rectangle(image, self.p1, self.p2,
                          MouseBoxDrawer.DEFAULT_BOX_COLOR, 2)
            cv2.circle(image, self.p1, 5,
                       MouseBoxDrawer.DEFAULT_BOX_EDGES_COLOR, -1)
            cv2.circle(image, self.p2, 5,
                       MouseBoxDrawer.DEFAULT_BOX_EDGES_COLOR, -1)

    def getCurrentData(self):
        if self.p1 != None and self.p2 != None:
            return {'type': self.namespace, 'rect': (self.p1, self.p2)}
        return super.getCurrentData()


class WSceneFrameVisualizer(WBaseWidget):

    def __init__(self, frame_steps=5):
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
        self.ui_spin_frame_step.setValue(frame_steps)

        self.ui_image.mousePressEvent = self.mousePressEvent
        self.ui_image.mouseMoveEvent = self.mouseMoveEvent
        self.ui_image.mouseReleaseEvent = self.mouseReleaseEvent

        # Interaction
        self.enable_interaction = False
        self.current_drawer = MouseBoxDrawer(self, 'BOX', self.drawerCallback)
        self.drawer_callbacks = []

    def enableInteraction(self, status=True):
        self.enable_interaction = status
        if self.current_drawer != None:
            self.current_drawer.reset()
        self.refresh()

    def addDrawerCallback(self, cb):
        self.drawer_callbacks.append(cb)

    def clearDrawerCallbacks(self):
        self.drawer_callbacks = []

    def drawerCallback(self, data):
        data['camera_pose'] = self.current_frame.getCameraPose()
        data['camera_matrix_inv'] = self.scene.camera_params.camera_matrix_inv
        data['camera_matrix'] = self.scene.camera_params.camera_matrix
        for c in self.drawer_callbacks:
            c(data)

    def mouseReleaseEvent(self, evt):
        if self.enable_interaction:
            if self.current_drawer != None:
                self.current_drawer.manageMouseRelease(evt)

    def mouseMoveEvent(self, evt):
        if self.enable_interaction:
            if self.current_drawer != None:
                self.current_drawer.manageMouseMove(evt)

        self.refresh()

    def mousePressEvent(self, evt):
        if self.enable_interaction:
            if self.current_drawer != None:
                self.current_drawer.manageClick(evt)

        self.refresh()

    def imageCoordinatesFromFrameCoordinates(self, x, y, tp=int):
        # TODO: move this function in utilities
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

    def drawLabel(self, image, label, color=colors.getColor('green')):
        height = 40
        width = 150
        cv2.rectangle(image, (0, 0), (width, height), color, -1)
        cv2.putText(image, label, (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def refresh(self):
        if self.scene:
            self.current_image_gizmo = self.current_image.copy()

            if self.enable_interaction:
                self.drawLabel(self.current_image_gizmo, "EDITABLE")

            if self.current_drawer != None:

                self.current_drawer.drawGizmo(self.current_image_gizmo)

            display_image = cv2.cvtColor(
                self.current_image_gizmo,
                cv2.COLOR_BGR2RGB
            )

            self.drawInstances(display_image)

            pix = PyQtImageConverter.cvToQPixmap(display_image)

            pix = pix.scaled(
                self.ui_image.size().width() - 5,
                self.ui_image.size().height() - 1,
                QtCore.Qt.KeepAspectRatio
            )

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

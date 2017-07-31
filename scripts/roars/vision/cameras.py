"""Cameras management."""
import cv2
import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv_bridge

from roars.rosutils.logger import Logger

import os
import sys
import yaml

#############################################################################
#############################################################################
#############################################################################


def yaml_opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat


yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", yaml_opencv_matrix)


def yaml_load_legacy(camera_file):
    with open(camera_file) as fin:
        c = fin.read()
        # some operator on raw conent of c may be needed
        c = "%YAML 1.1" + os.linesep + "---" + \
            c[len("%YAML:1.0"):] if c.startswith("%YAML:1.0") else c
        result = yaml.load(c)
        return result
    return None

#############################################################################
#############################################################################
#############################################################################


class Camera(object):

    def __init__(self, configuration_file):
        self.configuration_file = configuration_file

        yaml_cfg = yaml_load_legacy(configuration_file)

        try:
            self.width = int(yaml_cfg["image_width"])
            self.height = int(yaml_cfg["image_height"])
            self.fx = float(yaml_cfg["camera_matrix"][0][0])
            self.fy = float(yaml_cfg["camera_matrix"][1][1])
            self.cx = float(yaml_cfg["camera_matrix"][0][2])
            self.cy = float(yaml_cfg["camera_matrix"][1][2])
            self.k1 = float(yaml_cfg["distortion_coefficients"][0])
            self.k2 = float(yaml_cfg["distortion_coefficients"][1])
            self.p1 = float(yaml_cfg["distortion_coefficients"][2])
            self.p2 = float(yaml_cfg["distortion_coefficients"][3])

            print("Loaded Camera Parameters:",
                  self.fx,
                  self.fy,
                  self.cx,
                  self.cy,
                  self.k1,
                  self.k2,
                  self.p1,
                  self.p2
                  )
        except:
            print("CAMERA CALIBRATION IS HARDCODED!! MISSING TXT CONFIGURATION FILE:{}".format(
                configuration_file))

        self.camera_matrix = np.array([])
        self.camera_matrix_inv = np.array([])
        self.distortion_coefficients = np.array([])
        self.buildCameraMatrix()

    def buildCameraMatrix(self):
        self.camera_matrix = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.distortion_coefficients = np.array([
            self.k1, self.k2, self.p1, self.p2
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

    def getCameraFile(self):
        return self.configuration_file

    def get3DPoint(self, u, v, framergbd):
        d = float(framergbd.depth_image[u, v]) / float(framergbd.depth_scale)
        px = (d / self.fx) * (v - self.cx)
        py = (d / self.fy) * (u - self.cy)
        pz = d
        return np.array([px, py, pz])


############################################################
############################################################
############################################################
############################################################


class FrameRGBD(object):
    CV_BRIDGE = CvBridge()

    def __init__(self, rgb_image=None, depth_image=None, time=None):
        self.rgb_image = rgb_image
        self.depth_image = depth_image
        self.depth_scale = 1
        self.time = time
        if self.rgb_image == None or self.depth_image == None:
            self.valid = False
        else:
            self.valid = True

    def isValid(self):
        return self.valid

    def getPointCloud(self, camera, mask=None):
        points = []
        if self.isValid():
            for u in range(0, self.rgb_image.shape[0]):
                for v in range(0, self.rgb_image.shape[1]):
                    p = camera.get3DPoint(u, v, self)
                    points.append(p)
        return points

    @staticmethod
    def buildFromMessages(rgb_msg, depth_msg, depth_scale=1000):
        frame = FrameRGBD()
        frame.depth_scale = 1000
        frame.time = rgb_msg.header.stamp
        try:
            frame.rgb_image = FrameRGBD.CV_BRIDGE.imgmsg_to_cv2(
                rgb_msg, "bgr8")
        except CvBridgeError as e:
            Logger.error(e)
            return frame

        try:
            frame.depth_image = FrameRGBD.CV_BRIDGE.imgmsg_to_cv2(
                depth_msg, "16UC1")
        except CvBridgeError as e:
            Logger.error(e)
            return frame

        frame.valid = True
        return frame

############################################################
############################################################
############################################################
############################################################


class CameraRGBD(Camera):

    def __init__(self, configuration_file, rgb_topic, depth_topic, callback_buffer_size=1, approx_time_sync=0.1):
        super(CameraRGBD, self).__init__(
            configuration_file=configuration_file)
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.callback_buffer_size = callback_buffer_size
        self.approx_time_sync = approx_time_sync

        self.rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], callback_buffer_size, approx_time_sync)
        ts.registerCallback(self.topicCallback)

        self._user_callbacks = []

    def registerUserCallabck(self, callback):
        self._user_callbacks.append(callback)

    def topicCallback(self, rgb_msg, depth_msg):
        frame = FrameRGBD.buildFromMessages(
            rgb_msg, depth_msg)
        for c in self._user_callbacks:
            c(frame)


############################################################
############################################################
############################################################
############################################################


class FrameRGB(object):
    CV_BRIDGE = CvBridge()

    def __init__(self, rgb_image=None, time=None):
        self.rgb_image = rgb_image
        self.time = time
        if self.rgb_image == None:
            self.valid = False
        else:
            self.valid = True

    def isValid(self):
        return self.valid

    @staticmethod
    def buildFromMessages(rgb_msg):
        frame = FrameRGB()
        frame.time = rgb_msg.header.stamp
        try:
            frame.rgb_image = FrameRGBD.CV_BRIDGE.imgmsg_to_cv2(
                rgb_msg, "bgr8")
        except CvBridgeError as e:
            try:
                frame.rgb_image = FrameRGBD.CV_BRIDGE.imgmsg_to_cv2(
                    rgb_msg, "8UC1")
            except CvBridgeError as e:
                Logger.error(e)
                return frame

        frame.valid = True
        return frame

    @staticmethod
    def buildFromMessagesCompressed(rgb_msg):
        frame = FrameRGB()
        frame.time = rgb_msg.header.stamp

        np_arr = np.fromstring(rgb_msg.data, np.uint8)
        try:
            frame.rgb_image = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        except:
            try:
                frame.rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except:
                return frame

        frame.valid = True
        return frame

############################################################
############################################################
############################################################
############################################################


class CameraRGB(Camera):

    def __init__(self, configuration_file, rgb_topic, callback_buffer_size=1, compressed_image=False):
        super(CameraRGB, self).__init__(
            configuration_file=configuration_file)
        self.rgb_topic = rgb_topic
        self.callback_buffer_size = callback_buffer_size
        self.compressed_image = compressed_image

        if compressed_image:
            self.rgb_sub = rospy.Subscriber(
                self.rgb_topic, CompressedImage, self.topicCallback, queue_size=self.callback_buffer_size)
        else:
            self.rgb_sub = rospy.Subscriber(
                self.rgb_topic, Image, self.topicCallback, queue_size=self.callback_buffer_size)
        self._user_callbacks = []

    def registerUserCallabck(self, callback):
        self._user_callbacks.append(callback)

    def topicCallback(self, rgb_msg):
        if self.compressed_image:
            frame = FrameRGB.buildFromMessagesCompressed(rgb_msg)
        else:
            frame = FrameRGB.buildFromMessages(rgb_msg)
        for c in self._user_callbacks:
            c(frame)

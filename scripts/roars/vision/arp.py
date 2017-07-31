#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import PyKDL
import rospy
import tf
import cv2
import aruco
import numpy as np
import roars.geometry.transformations as transformations
from roars.vision.arucoutils import MarkerDetector
import roars.geometry.transformations as transformations
import json

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


class ARPConfiguration(object):

    def __init__(self, configuration_file=""):

        if configuration_file == "":
            self.options = ARPConfiguration.getPrototypeConfiguration()
        else:
            with open(configuration_file) as data_file:
                self.options = json.load(data_file)

        self.valid = False
        self.markers_map = {}
        self.markers_poses = {}
        if len(self.options) > 0:
            self.valid = True
            for id, data in self.options["markers"].iteritems():
                self.markers_map[int(id)] = data["size"]
                self.markers_poses[int(id)] = transformations.KDLFromArray(
                    data["pose"])

    @staticmethod
    def getPrototypeConfiguration():
        cube_offset = 0.087
        delta_cube_offset = 0.05
        component_markers_map = {
            "markers": {
                400: {"size": 0.04, "pose": [0, 0, 0.2215, 0, 0, 0]},
                401: {"size": 0.04, "pose": [-0.025, 0, cube_offset + 2 * delta_cube_offset, 0, -np.pi * 0.5, 0]},
                402: {"size": 0.04, "pose": [-0.025, 0, cube_offset + delta_cube_offset, 0, -np.pi * 0.5, 0]},
                403: {"size": 0.04, "pose": [-0.025, 0, cube_offset, 0, -np.pi * 0.5, 0]},

                410: {"size": 0.04, "pose": [0.0, 0.025, cube_offset + 2 * delta_cube_offset, 0, -np.pi * 0.5, -np.pi * 0.5]},
                411: {"size": 0.04, "pose": [0.0, 0.025, cube_offset + delta_cube_offset, 0, -np.pi * 0.5, -np.pi * 0.5]},
                412: {"size": 0.04, "pose": [0.0, 0.025, cube_offset, 0, -np.pi * 0.5, -np.pi * 0.5]},

                404: {"size": 0.04, "pose": [0.0, -0.025, cube_offset + 2 * delta_cube_offset, 0, np.pi * 0.5, -np.pi * 0.5]},
                405: {"size": 0.04, "pose": [0.0, -0.025, cube_offset + delta_cube_offset, 0, np.pi * 0.5, -np.pi * 0.5]},
                406: {"size": 0.04, "pose": [0.0, -0.025, cube_offset, 0, np.pi * 0.5, -np.pi * 0.5]},

                407: {"size": 0.04, "pose": [0.025, 0.0, cube_offset + 2 * delta_cube_offset, 0, np.pi * 0.5, 0]},
                408: {"size": 0.04, "pose": [0.025, 0.0, cube_offset + delta_cube_offset, 0, np.pi * 0.5, 0]},
                409: {"size": 0.04, "pose": [0.025, 0.0, cube_offset, 0, np.pi * 0.5, 0]}
            }
        }
        return component_markers_map

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


class ARP(PyKDL.Frame):

    def __init__(self, configuration_file="", camera_file=""):
        super(ARP, self).__init__()
        self.arp_configuration = ARPConfiguration(
            configuration_file=configuration_file
        )

        self.marker_detector = MarkerDetector(
            camera_file=camera_file,
            z_up=True
        )

    def detect(self, img, debug_draw=False):

        markers = self.marker_detector.detectMarkersMap(
            img,
            markers_map=self.arp_configuration.markers_map
        )

        mean_pose = [0, 0, 0, 0, 0, 0, 0]

        for id, marker in markers.iteritems():
            #⬢⬢⬢⬢⬢➤ Debug Draw
            if debug_draw:
                marker.draw(img)

            #⬢⬢⬢⬢⬢➤ Sum marker single contribute
            contribute = self.arp_configuration.markers_poses[id].Inverse()
            contribute = marker * contribute

            mean_pose[0] += contribute.p.x()
            mean_pose[1] += contribute.p.y()
            mean_pose[2] += contribute.p.z()
            qx, qy, qz, qw = contribute.M.GetQuaternion()
            mean_pose[3] += qx
            mean_pose[4] += qy
            mean_pose[5] += qz
            mean_pose[6] += qw

        mean_pose = np.array(
            [mean_pose],
            dtype=np.float32) / float(len(markers))

        mean_pose = mean_pose.reshape(7)
        mean_pose = transformations.KDLFromArray(mean_pose, fmt='XYZQ')

        if not math.isnan(mean_pose.p.x()):
            return mean_pose

        return None

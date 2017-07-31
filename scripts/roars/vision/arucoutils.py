"""Markers detection."""

import math
import PyKDL
import rospy
import tf
import cv2
import aruco
import numpy as np
import roars.geometry.transformations as transformations

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


class ARMarker(PyKDL.Frame):

    def __init__(self, aruco_marker=None, z_up=False):
        super(ARMarker, self).__init__()
        self.aruco_marker = aruco_marker
        self.z_up = z_up
        rot, _ = cv2.Rodrigues(self.aruco_marker.Rvec)

        self.M = PyKDL.Rotation(
            rot[0, 0], rot[0, 1], rot[0, 2],
            rot[1, 0], rot[1, 1], rot[1, 2],
            rot[2, 0], rot[2, 1], rot[2, 2]
        )
        self.p = PyKDL.Vector(
            self.aruco_marker.Tvec[0],
            self.aruco_marker.Tvec[1],
            self.aruco_marker.Tvec[2]
        )
        if self.z_up:
            self.M.DoRotX(-np.pi / 2.0)
            self.M.DoRotZ(-np.pi)

        self.corners = []
        for p in self.aruco_marker:
            self.corners.append(p)

        self.radius = int(
            0.5 * np.linalg.norm(self.corners[0] - self.corners[2]))
        self.center = np.array([0.0, 0.0])
        for p in self.corners:
            self.center += p
        self.center = self.center / 4

        self.side_in_pixel = 0
        for i in range(0, len(self.corners)):
            i_next = (i + 1) % len(self.corners)
            p1 = np.array([self.corners[i]])
            p2 = np.array([self.corners[i_next]])
            self.side_in_pixel += np.linalg.norm(p1 - p2)
        self.side_in_pixel = self.side_in_pixel / float(len(self.corners))

    def getID(self):
        return self.aruco_marker.id

    def getName(self):
        return "marker_{}".format(self.getID())

    def draw(self, image, color=np.array([255, 255, 255]), scale=1, draw_center=False):
        self.aruco_marker.draw(image, color, scale)
        if draw_center:
            cv2.circle(
                image,
                (int(self.center[0]), int(self.center[1])),
                radius=3,
                color=np.array([255, 255, 255]),
                thickness=3 * scale
            )

    def getPlaneCoefficients(self):
        return transformations.planeCoefficientsFromFrame(self)

    def get2DFrame(self):
        ax = np.array(self.corners[1] - self.corners[0])
        ax = ax / np.linalg.norm(ax)
        ay = np.array([-ax[1], ax[0]])
        return transformations.KDLFrom2DRF(ax, ay, self.center)

    def applyCorrection(self, frame):
        self_frame = PyKDL.Frame()
        self_frame.M = self.M
        self_frame.p = self.p
        self_frame = self_frame * frame
        self.M = self_frame.M
        self.p = self_frame.p

    def measurePixelRatio(self, side_in_meter):
        return side_in_meter / self.side_in_pixel

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################


class MarkerDetector(object):

    def __init__(self, camera_file, z_up=False):
        self.camparams = aruco.CameraParameters()
        self.camparams.readFromXMLFile(camera_file)
        self.aruco_detector = aruco.MarkerDetector()
        self.z_up = z_up

    def detectMarkers(self, image, markers_metric_size=-1.0, markers_map=None):
        markers = self.aruco_detector.detect(image)
        final_markers = []
        if markers_metric_size < 0 and markers_map == None:
            return final_markers

        for marker in markers:
            if markers_metric_size > 0:
                marker.calculateExtrinsics(markers_metric_size, self.camparams)
                final_markers.append(ARMarker(marker, self.z_up))
            elif markers_map != None:
                if marker.id in markers_map:
                    marker.calculateExtrinsics(
                        markers_map[marker.id], self.camparams)
                    final_markers.append(ARMarker(marker, self.z_up))

        return final_markers

    def detectMarkersMap(self, image, markers_metric_size=-1.0, markers_map=None):
        markers = self.detectMarkers(
            image, markers_metric_size, markers_map)
        markers_map = {}
        for marker in markers:
            markers_map[marker.getID()] = marker
        return markers_map

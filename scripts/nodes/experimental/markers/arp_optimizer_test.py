#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.cameras import Camera
from roars.vision.arucoutils import MarkerDetector
from roars.vision.arp import ARP
from roars.vision.augmentereality import BoundingBoxFromSixPoints, VirtualObject
import roars.geometry.transformations as transformations

import roars.vision.cvutils as cvutils
import cv2
import numpy as np
import os
import json
import glob
import sys
from scipy.optimize import minimize
import random
import math
import json
import PyKDL


class ArpBuilder(object):
    DEBUG_OPTIMIZITION_OUTPUT = True

    def __init__(self, markers_data, marker_size):
        self.markers_data = markers_data
        self.marker_size = marker_size

    def findBestPoses(self, markers_id_list, x0=np.array([-0.00, 0, 0.0, 0, 0, 0])):
        res_map = {"markers": {}}
        for id in markers_id_list:
            print("Optimizing marker:", id)
            res = self.findBestPose(id, x0)
            frame = transformations.KDLFromArray(res.x, fmt='RPY').Inverse()
            pose = transformations.KDLtoNumpyVector(frame, fmt='RPY')
            res_map["markers"][id] = {
                "pose": pose.reshape(6).tolist(),
                "size": self.marker_size
            }
            print pose
            # res_map[id] =
        return res_map

    def findBestPose(self, marker_id, x0=np.array([-0.00, 0, 0.0, 0, 0, 0])):
        frames = self.extractOnlyId(self.markers_data, marker_id)
        print ("Optimizing", marker_id, len(frames))
        res = minimize(
            self.commonPointFrames,
            x0,
            args=(frames),
            method='Powell',
            options={'xtol': 1e-6, 'disp': True, 'maxiter': 10000}
        )
        return res

    def computeClusterError(self, points):
        error = 0
        for i in range(0, len(points)):
            for j in range(0, len(points)):
                if i != j:
                    dist = points[i] - points[j]
                    dist = np.linalg.norm(dist)
                    error += dist * dist
        return math.sqrt(error)

    def commonPointFrames(self, x, frames):

        relative_frame = transformations.KDLFromArray(x, fmt='RPY')

        cumulation_points = []
        for frame in frames:
            computed_frame = frame * relative_frame.p

            img_points = cvutils.reproject3DPoint(
                computed_frame.x(),
                computed_frame.y(),
                computed_frame.z(),
                camera=camera
            )

            cumulation_points.append(img_points)

        error = self.computeClusterError(cumulation_points)
        if ArpBuilder.DEBUG_OPTIMIZITION_OUTPUT:
            print error
        return error

    def extractOnlyId(self, whole_data, id):
        frames = []
        for data in whole_data:
            if id in data:
                frames.append(data[id])
        return frames


#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("arp_optimizer_test")

#⬢⬢⬢⬢⬢➤ Sets HZ from parameters
node.setHz(node.setupParameter("hz", 30))
debug = node.setupParameter("debug", False)

camera_file = node.getFileInPackage(
    'roars',
    'data/camera_calibrations/microsoft_hd.yml'
)
camera = Camera(configuration_file=camera_file)


detections_folder = node.setupParameter(
    "detections_folder", "/home/daniele/Desktop/tmp/arp_calibration/markers_detections/")


max_number = 330

files = sorted(glob.glob(detections_folder + '*.txt'))

whole_data = []
for filename in files:
    basename = os.path.basename(filename)
    base = basename.split('.')[0]
    n = int(base.split('_')[1])
    data = np.loadtxt(filename)

    data_map = {}
    for d in data:
        try:
            id = int(d[0])
            frame = transformations.KDLFromArray(d[1:8], fmt='XYZQ')
            data_map[id] = frame
        except:
            pass

    whole_data.append(data_map)

    if n >= max_number:
        break
    print base, n

arp_builder = ArpBuilder(whole_data, 0.04)

ids = [400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412]
built_cfg = arp_builder.findBestPoses(ids, [0, 0, 0, 0, 0, 0])


f = open('prova.json', 'w')
f.write(json.dumps(built_cfg, indent=4))
f.close()

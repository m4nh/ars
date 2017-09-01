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
import PyKDL

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("arp_optimizer_validation")

#⬢⬢⬢⬢⬢➤ Sets HZ from parameters
node.setHz(node.setupParameter("hz", 30))
debug = node.setupParameter("debug", True)
output_folder = node.setupParameter(
    "output_folder", "/home/daniele/Desktop/tmp/arp_calibration/markers_detections")

if not os.path.exists(output_folder):
    print ("Path doesn't exist'")
    sys.exit(0)

camera_file = node.getFileInPackage(
    'roars',
    'data/camera_calibrations/microsoft_hd.yml'
)
camera = Camera(configuration_file=camera_file)

#⬢⬢⬢⬢⬢➤ ARP
arp_configuration = node.getFileInPackage(
    'roars',
    'data/arp_configurations/prototype_configuration.json'
)
arp = ARP(configuration_file=arp_configuration, camera_file=camera_file)


images_folder = node.setupParameter(
    "images_folder", "/home/daniele/Desktop/tmp/arp_calibration/usb_cam_image_raw_compressed")

if not images_folder.endswith('/'):
    images_folder = images_folder + '/'
print images_folder

files = sorted(glob.glob(images_folder + '*.jpg'))

for filename in files:
    basename = os.path.basename(filename)
    base = basename.split('.')[0]

    img = cv2.imread(filename)

    contributes = []
    arp_pose = arp.detect(img, debug_draw=True, contributes_output=None)

    if arp_pose:
        print arp_pose
        img_points = cvutils.reproject3DPoint(
            arp_pose.p.x(),
            arp_pose.p.y(),
            arp_pose.p.z(),
            camera=camera
        )

        cv2.circle(
            img,
            (int(img_points[0]), int(img_points[1])),
            5,
            (0, 0, 255),
            -1
        )

        rod_p = PyKDL.Frame(PyKDL.Vector(0, 0, 0.06))

        vob_rod = VirtualObject(
            arp_pose, size=[0.005, 0.005, rod_p.p.z()])
        vob_body = VirtualObject(
            arp_pose * rod_p, size=[0.05, 0.05, 0.22 - rod_p.p.z()])

        vob_rod.draw(img, camera=camera, color=(0, 255, 0))
        vob_body.draw(img, camera=camera, color=(0, 255, 255))

        print "Contributes", contributes
        for c in contributes:
            img_points = cvutils.reproject3DPoint(
                c.p.x(),
                c.p.y(),
                c.p.z(),
                camera=camera
            )
            cv2.circle(
                img,
                (int(img_points[0]), int(img_points[1])),
                4,
                (0, 255, 255),
                -1
            )

    if debug:
        cv2.imshow("img", np.flip(img, 0))
        c = cv2.waitKey(0)
        if c == 1048689:
            sys.exit(0)
    print basename

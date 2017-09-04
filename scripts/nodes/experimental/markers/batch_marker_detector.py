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

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("batch_marker_detector")

#⬢⬢⬢⬢⬢➤ Sets HZ from parameters
node.setHz(node.setupParameter("hz", 30))
debug = node.setupParameter("debug", False)
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

marker_detector = MarkerDetector(camera_file, z_up=True)

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
    markers = marker_detector.detectMarkersMap(img, 0.04)

    out_list = []
    for id, marker in markers.iteritems():
        if debug:
            marker.draw(img)
            vob = VirtualObject(marker, size=[0.04, 0.04, 0.01])
            vob.draw(img, camera=camera)

        out_list.append(
            np.hstack((np.array(id).reshape(1, 1), transformations.KDLtoNumpyVector(marker).reshape(1, 7))))

    s = len(out_list)
    out_list = np.array(out_list).reshape(s, 8)
    output_file = os.path.join(output_folder, base + ".txt")
    np.savetxt(output_file, out_list,
               fmt='%d %.8f %.8f %.8f %.8f %.8f %.8f %.8f')

    if debug:
        control = np.loadtxt(output_file)

        if control.shape[0] > 0:
            for c in control:
                m = transformations.KDLFromArray(c[1:8], fmt='XYZQ')
                vob2 = VirtualObject(m, size=[0.04, 0.04, 0.01])
                vob2.draw(img, camera=camera, color=(0, 255, 0))

    if debug:
        cv2.imshow("img", img)
        c = cv2.waitKey(0)
        if c == 1048689:
            sys.exit(0)
    print basename, len(markers)

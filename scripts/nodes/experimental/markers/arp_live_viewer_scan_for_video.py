#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.cameras import CameraRGB
from roars.vision.arucoutils import MarkerDetector
from roars.vision.arp import ARP
from roars.vision.augmentereality import BoundingBoxFromFourPoints, VirtualObject, BoundingBoxGenerator
from std_msgs.msg import Bool
import roars.geometry.transformations as transformations

import roars.vision.cvutils as cvutils
import cv2
import numpy as np
import os
import json
import glob
import sys
import PyKDL


collected_objects = [
    {
        'rf': [-0.0615924,   0.06422103,  0.64401128, 0.85372638, 0.39145536, -0.12949947, 0.31525226],
        'size': [0.05621781,  0.04149845,  0.07425442]
    },
    {
        'rf': [0.04838199,  0.06189513,  0.6455786,  0.9376154, 0.11573905, -0.03412503,  0.32530516],
        'size': [0.06787839,  0.04252589,  0.07461602]
    }
]


#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("arp_live_viewer")

#⬢⬢⬢⬢⬢➤ Sets HZ from parameters
node.setHz(node.setupParameter("hz", 30))
debug = node.setupParameter("debug", True)


#⬢⬢⬢⬢⬢➤ Creates Camera Proxy
camera_topic = node.setupParameter(
    "camera_topic",
    "/usb_cam/image_raw/compressed"
)
camera_file = node.getFileInPackage(
    'roars',
    'data/camera_calibrations/microsoft_hd_focus2.yml'
)
camera = CameraRGB(
    configuration_file=camera_file,
    rgb_topic=camera_topic,
    compressed_image="compressed" in camera_topic
)

#⬢⬢⬢⬢⬢➤ ARP
arp_configuration = node.getFileInPackage(
    'roars',
    'data/arp_configurations/prototype_configuration.json'
)
arp = ARP(configuration_file=arp_configuration, camera_file=camera_file)


camera_extrinsics = np.array([0.155488958419836, -0.001157008853558, -0.174804009533962, -
                              0.708182213284959, 0.705872162784811, 0.012869505975847, -0.007537798634064])

camera_extrinsics = transformations.NumpyVectorToKDL(camera_extrinsics)

first_camera_pose = None
windows_config = False

#⬢⬢⬢⬢⬢➤ Camera Callback


def cameraCallback(frame):
    global current_arp_pose, collected_boxes, first_camera_pose, windows_config
    if not windows_config:
        windows_config = True
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    #⬢⬢⬢⬢⬢➤ Grabs image from Frame
    img = frame.rgb_image.copy()

    camera_pose = node.retrieveTransform(
        "/comau_smart_six/link6", "/comau_smart_six/base_link", -1)
    if camera_pose == None:
        print("No camera pose available")
        return

    camera_pose = camera_pose * camera_extrinsics

    if first_camera_pose == None:
        first_camera_pose = camera_pose

    for o in collected_objects:
        print o
        rf = transformations.NumpyVectorToKDL(o['rf'])
        size = o['size']

        rf = first_camera_pose * rf

        vo = VirtualObject(frame=rf, size=size)
        vo.draw(img, camera=camera, camera_frame=camera_pose,
                color=(243, 150, 33))

        print "B", transformations.KDLtoNumpyVector(rf), size

    cv2.imshow("img", img)
    c = cv2.waitKey(1)
    if c == 32:
        pass


camera.registerUserCallabck(cameraCallback)

while node.isActive():
    node.tick()

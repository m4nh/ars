#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.cameras import CameraRGB
from roars.vision.arucoutils import MarkerDetector
from roars.vision.arp import ARP
import roars.vision.cvutils as cvutils
import cv2
import numpy as np
import os

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("rosnode_example")

#⬢⬢⬢⬢⬢➤ Sets HZ from parameters
node.setHz(node.setupParameter("hz", 30))

#⬢⬢⬢⬢⬢➤ Creates Camera Proxy
camera_topic = node.setupParameter(
    "camera_topic",
    "/camera/rgb/image_raw/compressed"
)
camera_file = node.getFileInPackage(
    'roars',
    'data/camera_calibrations/asus_xtion.yml'
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


#⬢⬢⬢⬢⬢➤ Camera Callback
def cameraCallback(frame):
    #⬢⬢⬢⬢⬢➤ Grabs image from Frame
    img = frame.rgb_image.copy()

    arp_pose = arp.detect(img, debug_draw=True)
    if arp_pose:
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

    #⬢⬢⬢⬢⬢➤ Show
    cv2.imshow("output", img)
    cv2.waitKey(1)


camera.registerUserCallabck(cameraCallback)

#⬢⬢⬢⬢⬢➤ Main Loop
while node.isActive():
    node.tick()

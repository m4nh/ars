#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.cameras import CameraRGB
from roars.vision.arucoutils import MarkerDetector
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

#⬢⬢⬢⬢⬢➤ Creates marker detector
marker_detector = MarkerDetector(camera_file=camera_file, z_up=True)

markers_map = {}
for i in range(1, 1024):
    markers_map[i] = 0.04


#⬢⬢⬢⬢⬢➤ Camera Callback
def cameraCallback(frame):
    #⬢⬢⬢⬢⬢➤ Grabs image from Frame
    img = frame.rgb_image.copy()

    #⬢⬢⬢⬢⬢➤ Detects markers
    markers = marker_detector.detectMarkersMap(
        frame.rgb_image, markers_map=markers_map)

    #⬢⬢⬢⬢⬢➤ draw markers
    for id, marker in markers.iteritems():
        marker.draw(img)
        node.broadcastTransform(
            marker,
            "marker_{}".format(id),
            "world",
            node.getCurrentTime()
        )

    #⬢⬢⬢⬢⬢➤ Show
    cv2.imshow("output", img)
    cv2.waitKey(1)


camera.registerUserCallabck(cameraCallback)

#⬢⬢⬢⬢⬢➤ Main Loop
while node.isActive():
    node.tick()

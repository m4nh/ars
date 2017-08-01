#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.cameras import CameraRGB
from roars.vision.arucoutils import MarkerDetector
from roars.vision.arp import ARP
from roars.vision.augmentereality import BoundingBoxFromSixPoints, VirtualObject
import roars.geometry.transformations as transformations

import roars.vision.cvutils as cvutils
import cv2
import numpy as np
import os
import json

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("ar_object_visualizer")

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

#⬢⬢⬢⬢⬢➤ Robot pose
robot_ee_tf_name = node.setupParameter(
    "robot_ee_tf_name", "/comau_smart_six/link6")
robot_world_tf_name = node.setupParameter(
    "robot_world_tf_name", "/comau_smart_six/base_link")
arp_robot_pose = transformations.KDLFromArray(
    [
        1.171241733010316866e+00, 2.038211444054240443e-02, 2.609647481922321988e-01,
        3.170776406284797622e-01,
        -2.224684033246829545e-03, 9.483374663308693497e-01, -1.062405513804032947e-02
    ],
    fmt='XYZQ'
)
robot_on_hand_transform = transformations.KDLFromArray(
    [
        0.09131591676464931, 0.023330268359173824, - 0.19437327402734972,
        -0.7408449656427065, 0.7505081899194715, 0.01462135745716728, - 0.01655531561119271
        # 0.109010740854854,	0.0194781279660590,	-0.299749509111702,
        # -0.698978847276796, 0.707540085021683, 0.0743762216760345, -0.0726895920769237
    ],
    fmt='XYZQ'
)
arp_camera_pose = arp_robot_pose * robot_on_hand_transform


#⬢⬢⬢⬢⬢➤ Points storage
points_per_object = node.setupParameter("points_per_object", 6)
points_file = node.setupParameter(
    "output_file", "/home/daniele/Desktop/temp/scan5_points.json")
object_points = []
with open(points_file, 'r') as handle:
    object_points = json.load(handle)

#⬢⬢⬢⬢⬢➤ Creates Virtual Objects
virtual_objects = []
for obj in object_points:
    virtual_object = BoundingBoxFromSixPoints(obj)
    arp_vo = virtual_object.buildVirtualObject()
    size = arp_vo.size
    label = arp_vo.label
    arp_vo = arp_camera_pose * arp_vo
    virtual_objects.append(VirtualObject(frame=arp_vo, size=size, label=label))

print(virtual_objects)


#⬢⬢⬢⬢⬢➤ Camera Callback
def cameraCallback(frame):
    #⬢⬢⬢⬢⬢➤ Grabs image from Frame
    img = frame.rgb_image.copy()

    robot_pose = node.retrieveTransform(
        robot_ee_tf_name,
        robot_world_tf_name,
        -1
    )

    if robot_pose:
        camera_pose = robot_pose * robot_on_hand_transform
        print("Pose ready!", camera_pose)

        for vo in virtual_objects:
            vo.draw(img, camera=camera, camera_frame=camera_pose)

    #⬢⬢⬢⬢⬢➤ Show
    cv2.imshow("output", img)
    c = cv2.waitKey(1)
    if c == 113:
        node.close()


camera.registerUserCallabck(cameraCallback)

#⬢⬢⬢⬢⬢➤ Main Loop
while node.isActive():
    node.tick()

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


collected_points = []
collected_boxes = []
current_arp_pose = None


def newPointCallback(msg):
    global collected_points, current_arp_pose, collected_boxes
    if current_arp_pose:
        point_3d = np.array([
            current_arp_pose.p.x(),
            current_arp_pose.p.y(),
            current_arp_pose.p.z()
        ])
        collected_points.append(point_3d)
        print("Collected points", len(collected_points))
        if len(collected_points) == 4:
            box = BoundingBoxFromFourPoints(collected_points)
            collected_boxes.append(box)
            collected_points = []



#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("arp_live_viewer")

#⬢⬢⬢⬢⬢➤ Sets HZ from parameters
node.setHz(node.setupParameter("hz", 30))
debug = node.setupParameter("debug", True)

new_point_publisher = node.createPublisher("~new_point_event", Bool)
new_point_publisher_callback = node.createSubscriber(
    "~new_point_event", Bool, newPointCallback)


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

windows_config = False

#⬢⬢⬢⬢⬢➤ Camera Callback


def cameraCallback(frame):
    global current_arp_pose, collected_boxes, windows_config
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

    arp_pose = arp.detect(img, debug_draw=False, contributes_output=None)
    if arp_pose:
        current_arp_pose = arp_pose

        rod_p = PyKDL.Frame(PyKDL.Vector(0, 0, 0.06))
        vob_rod = VirtualObject(
            arp_pose, size=[0.005, 0.005, rod_p.p.z()])
        vob_body = VirtualObject(
            arp_pose * rod_p, size=[0.05, 0.05, 0.22 - rod_p.p.z()])

        vob_rod.draw(img, camera=camera, color=(54, 67, 244), thickness=1)
        vob_body.draw(img, camera=camera, color=(54, 67, 244), thickness=2)

    for p in collected_points:
        center = cvutils.reproject3DPoint(
            p[0], p[1], p[2], camera=camera, camera_pose=PyKDL.Frame())
        cv2.circle(img, center, 5, (243, 150, 33), -1)

    for b in collected_boxes:
        vo = b.buildVirtualObject()
        vo.draw(img, camera=camera, color=(243, 150, 33))
        print "B", transformations.KDLtoNumpyVector(vo), b.size

    cv2.imshow("img", img)
    c = cv2.waitKey(1)
    if c == 32:
        evt = Bool()
        evt.data = True
        new_point_publisher.publish(evt)


camera.registerUserCallabck(cameraCallback)

while node.isActive():
    node.tick()

# for filename in files:
#     basename = os.path.basename(filename)
#     base = basename.split('.')[0]

#     img = cv2.imread(filename)

#     contributes = []
#     arp_pose = arp.detect(img, debug_draw=True, contributes_output=None)

#     if arp_pose:
#         print arp_pose
#         img_points = cvutils.reproject3DPoint(
#             arp_pose.p.x(),
#             arp_pose.p.y(),
#             arp_pose.p.z(),
#             camera=camera
#         )

#         cv2.circle(
#             img,
#             (int(img_points[0]), int(img_points[1])),
#             5,
#             (0, 0, 255),
#             -1
#         )

#         rod_p = PyKDL.Frame(PyKDL.Vector(0, 0, 0.06))

#         vob_rod = VirtualObject(
#             arp_pose, size=[0.005, 0.005, rod_p.p.z()])
#         vob_body = VirtualObject(
#             arp_pose * rod_p, size=[0.05, 0.05, 0.22 - rod_p.p.z()])

#         vob_rod.draw(img, camera=camera, color=(0, 255, 0))
#         vob_body.draw(img, camera=camera, color=(0, 255, 255))

#         print "Contributes", contributes
#         for c in contributes:
#             img_points = cvutils.reproject3DPoint(
#                 c.p.x(),
#                 c.p.y(),
#                 c.p.z(),
#                 camera=camera
#             )
#             cv2.circle(
#                 img,
#                 (int(img_points[0]), int(img_points[1])),
#                 4,
#                 (0, 255, 255),
#                 -1
#             )

#     if debug:
#         cv2.imshow("img", np.flip(img, 0))
#         c = cv2.waitKey(0)
#         if c == 1048689:
#             sys.exit(0)
#     print basename

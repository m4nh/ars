#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.cameras import CameraRGB
from roars.vision.arp import ARP
import roars.geometry.transformations as transformations
import roars.vision.cvutils as cvutils
import cv2
import numpy as np
import os
import json

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

#⬢⬢⬢⬢⬢➤ Points storage
points_per_object = node.setupParameter("points_per_object", 6)
collected_points = []
output_file = node.setupParameter("output_file", "/tmp/arp_objects.json")

#⬢⬢⬢⬢⬢➤ Robot pose
use_world_coordinates = node.setupParameter("use_world_coordinates", True)
if use_world_coordinates:
    robot_ee_tf_name = node.setupParameter(
        "robot_ee_tf_name", "/comau_smart_six/link6")
    robot_world_tf_name = node.setupParameter(
        "robot_world_tf_name", "/comau_smart_six/base_link")
    camera_extrinsics_file = node.setupParameter(
        "camera_extrinsics_file", "/home/daniele/Desktop/datasets/roars_2017/indust/camera_extrinsics.txt")
    robot_to_camera_pose = transformations.KDLFromArray(
        np.loadtxt(camera_extrinsics_file), fmt='XYZQ')


#⬢⬢⬢⬢⬢➤ Splice array in sub arrays
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


#⬢⬢⬢⬢⬢➤ Save Output
def saveOutput():
    global collected_points, points_per_object
    probable_objects = list(chunks(collected_points, points_per_object))
    objects = []

    for o in probable_objects:
        if len(o) == points_per_object:
            objects.append(o)

    arp_output = {
        "points_per_objects": points_per_object,
        "objects": len(objects),
        "objects_points": objects
    }
    print("Saving ", output_file)
    with open(output_file, 'w') as handle:
        handle.write(json.dumps(arp_output, indent=4))


#⬢⬢⬢⬢⬢➤ Camera Callback
def cameraCallback(frame):
    #⬢⬢⬢⬢⬢➤ Grabs image from Frame
    img = frame.rgb_image.copy()

    if use_world_coordinates:
        robot_pose = node.retrieveTransform(
            robot_ee_tf_name,
            robot_world_tf_name,
            -1
        )
        if robot_pose == None:
            print("Robot pose not ready!")
            return
        else:
            camera_pose = robot_pose * robot_to_camera_pose

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
    c = cv2.waitKey(1)
    if c == 113:
        saveOutput()
        node.close()
    if c == 32 and arp_pose != None:

        if use_world_coordinates:
            arp_pose = camera_pose * arp_pose

        print("New Point Added", arp_pose.p)
        collected_points.append([
            arp_pose.p.x(), arp_pose.p.y(), arp_pose.p.z()
        ])
        if len(collected_points) % points_per_object == 0:
            print("New Object Stored")


camera.registerUserCallabck(cameraCallback)

#⬢⬢⬢⬢⬢➤ Main Loop
while node.isActive():
    node.tick()

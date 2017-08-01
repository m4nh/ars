#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.datasets.datasetutils import TrainingScene
from roars.rosutils.rosnode import RosNode

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("generate_roars_dataset_manifest")

scene_path = node.setupParameter("scene_path", '')
image_topic_name = node.setupParameter("image_topic_name", '')
robot_pose_name = node.setupParameter("robot_pose_name", '')
camera_intrisics_file = node.setupParameter("camera_intrisics_file", '')
camera_extrinsics_file = node.setupParameter("camera_extrinsics_file", '')
output_manifest_file = node.setupParameter("output_manifest_file", '')


#⬢⬢⬢⬢⬢➤ Create Scenes
scene = TrainingScene(
    scene_path=scene_path,
    image_topic_name=image_topic_name,
    robot_pose_name=robot_pose_name,
    camera_intrisics_file=camera_intrisics_file,
    camera_extrinsics_file=camera_extrinsics_file
)

#⬢⬢⬢⬢⬢➤ Initialize Scene, may fail
scene.initialize()

#⬢⬢⬢⬢⬢➤ Save Scene to file if it is valid
if scene.isValid():
    if output_manifest_file != '':
        scene.save(output_manifest_file)
        print("Saved!")
    else:
        print("Output File Name '{}' is not valid".format(output_manifest_file))
else:
    print("Scene is not valid!")

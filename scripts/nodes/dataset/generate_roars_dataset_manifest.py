#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.datasets.datasetutils import TrainingScene
from roars.rosutils.rosnode import RosNode

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("generate_roars_dataset_manifest")

scene_path = node.setupParameter("scene_path", '')
images_path = node.setupParameter("images_path", '')
robot_pose_name = node.setupParameter("robot_pose_name", '')
camera_intrisics_file = node.setupParameter("camera_intrisics_file", '')
camera_extrinsics_file = node.setupParameter("camera_extrinsics_file", '')
output_manifest_file = node.setupParameter("output_manifest_file", '')
force_relative = node.setupParameter("force_relative", False)

#⬢⬢⬢⬢⬢➤ Create Scenes
scene = TrainingScene(
    scene_path=scene_path,
    images_path=images_path,
    robot_pose_name=robot_pose_name,
    camera_intrisics_file=camera_intrisics_file,
    camera_extrinsics_file=camera_extrinsics_file
)

#⬢⬢⬢⬢⬢➤ Initialize Scene, may fail
scene.initialize()

#⬢⬢⬢⬢⬢➤ Save Scene to file if it is valid
if scene.isValid():
    if output_manifest_file != '':
        scene.save(output_manifest_file, force_relative=force_relative)
        print("Saved!")
    else:
        print("Output File Name '{}' is not valid".format(output_manifest_file))
else:
    print("Scene is not valid!")

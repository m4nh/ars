#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.datasets.datasetutils import TrainingScene, TrainingClassesMap
import sys
import argparse


ap = argparse.ArgumentParser()
###########################
ap.add_argument("--scene_path",
                required=True,
                help="Data folder")
###########################
ap.add_argument("--images_path",
                default='images',
                type=str,
                help="Images subfolder name")
###########################
ap.add_argument("--depths_path",
                default='depth',
                type=str,
                help="Depth Images subfolder name")
###########################
ap.add_argument("--camera_poses",
                default="robot_poses.txt",
                type=str,
                help="Relative filename in datafolder contanining camera poses")
###########################
ap.add_argument("--camera_intrinsics_file",
                default=TrainingScene.DEFAULT_CAMERA_PARAMS_NAME,
                type=str,
                help="Relative filename in datafolder contanining camera intrinsics")
###########################
ap.add_argument("--camera_extrinsics_file",
                default=TrainingScene.DEFAULT_CAMERA_POSE_NAME,
                type=str,
                help="Relative filename in datafolder contanining camera extrinsics (relative to Robot Wrist, if any")
###########################
ap.add_argument("--output_manifest_file",
                required=True,
                type=str,
                help="Output manifest filename")
###########################
ap.add_argument("--force_relative",
                default=True,
                type=bool,
                help="If TRUE the manifest contains relative paths. Thus, if the manifest is in the parent folder of 'Data Path', all can be moved safely")
###########################
ap.add_argument("--classes",
                required=True,
                type=str,
                help="List of dataset classes (';' separated)")
###########################
ap.add_argument("--force_no_classes",
                default=False,
                type=str,
                help="What is this??")

args = vars(ap.parse_args())

scene_path = args['scene_path']
images_path = args['images_path']
depth_path = args['depths_path']
robot_pose_name = args['camera_poses']
camera_intrisics_file = args['camera_intrinsics_file']
camera_extrinsics_file = args['camera_extrinsics_file']
output_manifest_file = args['output_manifest_file']
force_relative = args['force_relative']
classes = args['classes'].split(";")
force_no_classes = args['force_no_classes']

if not force_no_classes and len(classes) == 0:
    print("No classes found!")
    sys.exit(0)

# ⬢⬢⬢⬢⬢➤ Create Scenes
scene = TrainingScene(
    scene_path=scene_path,
    images_path=images_path,
    images_depth_path=depth_path,
    robot_pose_name=robot_pose_name,
    camera_intrisics_file=camera_intrisics_file,
    camera_extrinsics_file=camera_extrinsics_file
)

# ⬢⬢⬢⬢⬢➤ Initialize Scene, may fail
scene.initialize()

# ⬢⬢⬢⬢⬢➤ Save Scene to file if it is valid
if scene.isValid():

    cm = TrainingClassesMap(classes)
    scene.setClasses(cm)

    if output_manifest_file != '':
        scene.save(output_manifest_file, force_relative=force_relative)
        print("Saved!")
    else:
        print("Output File Name '{}' is not valid".format(output_manifest_file))
else:
    print("Scene is not valid!")

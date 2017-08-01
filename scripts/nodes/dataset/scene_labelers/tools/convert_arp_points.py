#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.augmentereality import BoundingBoxGenerator
from roars.datasets.datasetutils import TrainingInstance, JSONHelper
import roars.geometry.transformations as transformations
import os
import json
import sys

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("convert_arp_points")

#⬢⬢⬢⬢⬢➤ Output File
output_file = node.setupParameter('output_file', '')
if output_file == '':
    print("Invalid output file ''".format(output_file))
    sys.exit(0)

#⬢⬢⬢⬢⬢➤ Object Points
objects_points_file = node.setupParameter('objects_points_file', '')

if os.path.exists(objects_points_file):
    objects_data = JSONHelper.loadFromFile(objects_points_file)

    #⬢⬢⬢⬢⬢➤ Creates a Box Generator based on number of points
    number_of_points = objects_data["points_per_objects"]
    box_generator = BoundingBoxGenerator.getGenerator(number_of_points)

    if box_generator != None:

        #⬢⬢⬢⬢⬢➤ Generates frame for each object
        objects_frames = []
        for points in objects_data["objects_points"]:
            box = box_generator(points)
            inst = TrainingInstance(frame=box.getRF(), size=box.getSize())
            objects_frames.append(inst)

        #⬢⬢⬢⬢⬢➤ Write output
        objects_data["objects_instances"] = objects_frames
        JSONHelper.saveToFile(output_file, objects_data)

    else:
        print("No generator found for '{}' points:".format(number_of_points))

else:
    print("File '{}' doesn't exist".format(objects_points_file))

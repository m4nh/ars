#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.cameras import CameraRGB
from roars.vision.arucoutils import MarkerDetector
from roars.vision.augmentereality import VirtualObject
import roars.geometry.transformations as transformations
import cv2
import numpy as np
import os
import glob
from shutil import copyfile
import shutil
import collections


def convertLabels(img_file, label_file, img_size, class_map):
    content = []
    with open(label_file) as f:
        content = f.readlines()

    width = img_size[1]
    height = img_size[0]

    output_labels = []
    if len(content) > 0:
        n = int(content[0])
        for i in range(0, n):
            x, y, x2, y2, cl = content[i + 1].split(" ")
            w = float((float(x2) - float(x)) / width)
            h = float((float(y2) - float(y)) / height)
            x = float(float(x) / width + w * 0.5)
            y = float(float(y) / height + h * 0.5)

            cl = class_map[cl.replace("\n", "")]
            output_labels.append((cl, x, y, w, h))
            # img = cv2.imread(img_file)

            pw = int(w * width)
            ph = int(h * height)
            px = int(x * width) - int(pw * 0.5)
            py = int(y * height) - int(ph * 0.5)

            # print(i, (px, py, pw, ph, cl))
            # cv2.rectangle(img, (px, py), (px + pw, py + ph), (255), 1)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
    return np.array(output_labels)


def copyOutput(elements, output_folder, filters, negative=True):

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    output_folder_images = os.path.join(output_folder, "images")
    output_folder_labels = os.path.join(output_folder, "labels")
    output_folder_ids = os.path.join(output_folder, "ids")

    os.mkdir(output_folder)
    os.mkdir(output_folder_images)
    os.mkdir(output_folder_labels)
    os.mkdir(output_folder_ids)

    id_map = {}
    counter = 0
    for el in elements:

        for index in range(0, len(el["images"])):
            img = el["images"][index]
            label = el["labels"][index]
            id = el["ids"][index]

            scene_name = id.split("!")[0]
            internal_image = id.split("!")[1]

            if negative:
                if scene_name not in filters:
                    continue
            else:
                if scene_name in filters:
                    continue

            counter_string = '{}'.format(str(counter).zfill(5))

            output_image = os.path.join(
                output_folder_images, counter_string + ".jpg")
            output_label = os.path.join(
                output_folder_labels, counter_string + ".txt")
            output_id = os.path.join(
                output_folder_ids, counter_string + ".txt")

            copyfile(img, output_image)

            new_labels = convertLabels(img, label, (480, 640), class_map)
            np.savetxt(output_label, new_labels,
                       fmt="%d %1.4f %1.4f %1.4f %1.4f")

            f = open(output_id, 'w')
            f.write(id + "\n")

            if scene_name not in id_map:
                id_map[scene_name] = 0
            id_map[scene_name] = id_map[scene_name] + 1

            #print(counter_string, img, label, id)
            counter = counter + 1

    return collections.OrderedDict(sorted(id_map.items()))


#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("manual_labels_fusion")

dataset_folder = node.setupParameter(
    "dataset_folder", "/home/daniele/Desktop/datasets/roars_manual_2017/original_tolabel")

output_folder_test = node.setupParameter(
    "output_folder", "/home/daniele/Desktop/datasets/roars_manual_2017/manual_dataset_test")
output_folder_train = node.setupParameter(
    "output_folder", "/home/daniele/Desktop/datasets/roars_manual_2017/manual_dataset_train")


class_map = {
    "comp1": 0,
    "comp2": 1,
    "comp3": 2,
    "comp4": 3,
    "comp5": 4,
    "comp6": 5,
    "comp7": 6
}

elements = []
for file in glob.glob(dataset_folder + "/*.txt"):
    name = os.path.basename(file).split(".")[0]
    img_folder = os.path.join(os.path.dirname(file), name + "_images")
    labels_folder = os.path.join(os.path.dirname(file), name + "_labels")
    f = open(file, 'r')

    element = {
        "file": file,
        "img_folder": img_folder,
        "labels_folder": labels_folder,
        "name": name,
        "ext": os.path.basename(file).split(".")[1],
        "images": sorted(glob.glob(img_folder + "/*.jpg")),
        "labels": sorted(glob.glob(labels_folder + "/*.txt")),
        "ids": f.readlines()
    }
    elements.append(element)


id_map = copyOutput(elements, output_folder_train, [
                    'indust_scene_4_top', 'indust_scene_4_dome', 'smartphone'], False)
id_map = copyOutput(elements, output_folder_test, [
                    'indust_scene_4_top', 'indust_scene_4_dome', 'smartphone'], True)

for k, v in id_map.iteritems():
    print(k, v)

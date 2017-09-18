#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.cameras import CameraRGB
from roars.vision.arucoutils import MarkerDetector
from roars.vision.augmentereality import VirtualObject
import roars.geometry.transformations as transformations
from roars.detections.prediction import prediction
from roars.datasets.datasetutils import TrainingClass
import cv2
import numpy as np
import os
import glob

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("make_video_from_predictions")


#⬢⬢⬢⬢⬢➤ Sets HZ from parameters
node.setHz(node.setupParameter("hz", 30))

dataset_folder = node.setupParameter("dataset_folder", "")
predictions_name = node.setupParameter("predictions_name", "")


images_folder = node.setupParameter("images_folder", "")
predictions_folder = node.setupParameter("predictions_folder", "")

print "Loading", images_folder

images_files = sorted(glob.glob(images_folder + "/*.jpg"))
predictions_files = sorted(glob.glob(predictions_folder + "/*.txt"))
min_th = 0.2
cv2.namedWindow("img", cv2.WINDOW_NORMAL)

tick = 3
header_size = 80
hedaer_width = 450
names = [
    "component1",
    "component2",
    "component3",
    "component4",
    "component5",
    "component6",
    "component7"
]

for i in range(0, len(images_files), 10):
    print images_files[i], predictions_files[i]
    img = cv2.imread(images_files[i])
    for k in range(0, 2):
        img = cv2.pyrUp(img)

    print(img.shape)
    predictions_raw = np.loadtxt(predictions_files[i])

    for p in predictions_raw:
        if p[5] < min_th:
            continue
        pred = prediction.fromArray(p, centers=True)
        x = p[1] * img.shape[1]
        y = p[2] * img.shape[0]
        w = p[3] * img.shape[1]
        h = p[4] * img.shape[0]

        index = int(p[0])
        color = TrainingClass.getColorByLabel(index, output_type="RGB")

        p1 = (int(x - w * 0.5), int(y - h * 0.5))
        p2 = (int(x + w * 0.5), int(y + h * 0.5))
        print p1, p2
        cv2.rectangle(img, p1, p2, color, tick)
        cv2.rectangle(img, (p1[0], p1[1] - header_size),
                      (p1[0] + hedaer_width, p1[1]), color, -1)
        cv2.rectangle(img, (p1[0], p1[1] - header_size),
                      (p1[0] + hedaer_width, p1[1]), color, tick)

        cv2.putText(img, names[index], (p1[0], p1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)

    cv2.imshow("img", img)
    c = cv2.waitKey(1)
    if c == 13 or c == 113:
        break

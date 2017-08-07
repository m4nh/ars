#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.cameras import CameraRGB
from roars.detections.prediction import prediction
from roars.detections.tensorflow_detector_wrapper import tensorflow_detector_wrapper
from roars.gui import cv_show_detection
import cv2
import numpy as np
import argparse
from random import randint

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Load an inference graph and use it on the input images")
    parser.add_argument('-g','--graph',help="path to the pb file with the graph and weight definition",required=True)
    parser.add_argument('-l','--labelMap',help="path to the pbtxt containing the label definition",required=True)
    parser.add_argument('-v','--visualization',help="flag to enable visualization (only for debug)",action='store_true')
    args = parser.parse_args()

    detector = tensorflow_detector_wrapper(args.graph,args.labelMap)
    classes = detector.getClassDictionary()
    c_map = cv_show_detection.getColorMap(classes)

    #⬢⬢⬢⬢⬢➤ NODE  
    def image_callback(frame):
        #grab image from frame
        img = frame.rgb_image.copy()

        #convert to rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #detection
        detections = detector.detect(img_rgb)
        
        if args.visualization:
            immy = cv_show_detection.draw_prediction(img_rgb,detections,c_map)
            cv2.imshow('detections',immy)
            cv2.waitKey(1)

            #publish detection somewhere?

    #node creation
    node = RosNode("tf_obj_detector")

    #⬢⬢⬢⬢⬢➤ Sets HZ from parameters
    node.setHz(node.setupParameter("hz", 60))

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


    camera.registerUserCallabck(image_callback)

    #⬢⬢⬢⬢⬢➤ Main Loop
    while node.isActive():
        node.tick()

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
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Load an inference graph and use it on the input images")
    parser.add_argument('-g','--graph',help="path to the pb file with the graph and weight definition",required=True)
    parser.add_argument('-l','--labelMap',help="path to the pbtxt containing the label definition",required=True)
    parser.add_argument('-v','--visualization',help="flag to enable visualization (only for debug)",action='store_true')
    args = parser.parse_args()

    #create tensorflow wrapper class
    detector = tensorflow_detector_wrapper(args.graph,args.labelMap)
    if args.visualization:
        classes = detector.getClassDictionary()
        c_map = cv_show_detection.getColorMap(classes)

    #node creation
    node = RosNode("tf_obj_detector")

    #⬢⬢⬢⬢⬢➤ Sets HZ from parameters
    node.setHz(node.setupParameter("hz", 60))

    #create publisher
    prediction_topic = node.setupParameter(
        "prediction_topic",
        "/detections"
    )
    publisher = node.createPublisher(prediction_topic,numpy_msg(Floats))


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

    #⬢⬢⬢⬢⬢➤ NODE  
    def image_callback(frame):
        #grab image from frame
        img = frame.rgb_image.copy()

        #convert to rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #detection
        predictions = detector.detect(img_rgb)

        #convert predictions in a matrx with each row representing a different detection
        msg = prediction.toMatrix(predictions)
        
        #publish the detction
        #TODO: add timestamp to stay in sincro with frames?
        publisher.publish(msg)
        
        if args.visualization:
            immy = cv_show_detection.draw_prediction(img_rgb,predictions,c_map)
            cv2.imshow('detections',immy)
            cv2.waitKey(1)


    camera.registerUserCallabck(image_callback)

    #⬢⬢⬢⬢⬢➤ Main Loop
    while node.isActive():
        node.tick()

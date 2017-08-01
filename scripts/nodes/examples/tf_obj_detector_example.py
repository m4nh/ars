#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode
from roars.vision.cameras import CameraRGB
import cv2
import numpy as np
import os
import tensorflow as tf 
import argparse
from random import randint

def setup_inference_graph(path_to_pb):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_pb, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def random_color():
    return (randint(0,255),randint(0,255),randint(0,255))

def load_label_map(path_to_pbtxt):
    with open(path_to_pbtxt) as f_in:
        lines = f_in.readlines()
    
    current_id=-1
    label_map={}
    label_map[0]={'name':'Backgorund','id':0,'color':random_color()}
    for l in lines:
        if 'id' in l:
            current_id = int(l.split(':')[-1])
        if 'name' in l:
            class_name=l.split(':')[-1].strip().replace("'","")
            label_map[current_id]={'name':class_name,'id':current_id,'color':random_color()}

    print("label_map:", label_map)
    return label_map

def draw_detection(frame,boxes,classes,scores,label_map,use_normalized_coordinates=True,min_score_th=0.5,line_thickness=3,font_scale=0.7):
    assert boxes.shape[0]==classes.shape[0] and classes.shape[0]==scores.shape[0]
    for i,s in enumerate(scores):
        #draw box only if score is above th
        if s>min_score_th:
            current_class_id = classes[i]
            current_class = label_map[current_class_id]
            ymin, xmin, ymax, xmax = boxes[i]
            if use_normalized_coordinates:
                ymin = int(ymin*frame.shape[0])
                ymax = int(ymax*frame.shape[0])
                xmin = int(xmin*frame.shape[1])
                xmax = int(xmax*frame.shape[1])

            text_to_display = "{} - score:{:.2f}".format(current_class['name'],s)
            cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),current_class['color'],line_thickness)
            cv2.putText(frame,text_to_display,(xmin,ymin),cv2.FONT_HERSHEY_SIMPLEX,font_scale,255,2)
    return frame



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Load an inference graph and use it on the input images")
    parser.add_argument('-g','--graph',help="path to the pb file with the graph and weight definition",required=True)
    parser.add_argument('-l','--labelMap',help="path to the pbtxt containing the label definition",required=True)
    parser.add_argument('-v','--visualization',help="flag to enable visualization",action='store_true')
    args = parser.parse_args()

    for path in [args.graph,args.labelMap]:
        if not os.path.exists(path):
            print('ERROR: Unable to find {}'.format(path))
            exit()
    
    print('setting up graph')
    detection_graph = setup_inference_graph(args.graph)

    print('Load image labels')
    label_map = load_label_map(args.labelMap)

    #⬢⬢⬢⬢⬢➤ NODE
    with detection_graph.as_default():
        sess = tf.Session(graph=detection_graph)
        
        def image_callback(frame):
            #grab image from frame
            img = frame.rgb_image.copy()

            #convert to rgb
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype(np.uint8)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            img_rgb = np.expand_dims(img_rgb, axis=0)

            #fetch tensorflow ops
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: img_rgb})

            #eventually visualize
            if args.visualization:
                img=draw_detection(img,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),label_map,use_normalized_coordinates=True,line_thickness=8)
                cv2.imshow('detection',img)
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

import os
from roars.detections import prediction
import numpy as np

def check_existance(f):
    """
    Check if a file exist and do some nice print 
    """
    if not os.path.exists(f):
        print('ERROR: Unable to find file: {}'.format(f))
        return False
    return True

def read_predictions(file_name):
    """
    Read the annotation saved in yolo format in a .txt file and convert them in an array of detection objects
    """
    result=[]
    with open(file_name,'r') as f_in:
        lines=f_in.readlines()
    
    for l in lines:
        encoded = [float(f) for f in l.strip().split()]
        if len(encoded)==5:
            #confidence is missing, add it
            encoded+=[1]
        result.append(prediction.prediction.fromArray(encoded,centers=True))  
    return result

def associate(predicted_box,gt_boxes,use_iou=True):
    """
    returns the index of the ground truth box showing the largest overlap with the predicted one, if use_iou is True use iou to find the best matching box
    """
    intersections = [predicted_box.intersectionArea(bb) for bb in gt_boxes]
    if use_iou:
        p_area=predicted_box.getArea()
        intersections = [ii/(bb.getArea()+p_area-ii) for ii,bb in zip(intersections,gt_boxes)]
    gt_indx = np.argmax(intersections)
    return gt_indx, intersections[gt_indx]

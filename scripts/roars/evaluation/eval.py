import os
import cv2
import argparse
import glob
import numpy as np
from roars.detections import prediction
from roars.gui import cv_show_detection

CONFIDENCE_TH=[0.001+0.05*i for i in range(20)]
def check_existance(f):
    if not os.path.exists(f):
        print('ERROR: Unable to find folder: {}'.format(f))
        exit()

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

def associate(predicted_box,gt_boxes):
    intersections = [predicted_box.intersectionArea(bb) for bb in gt_boxes]
    gt_indx = np.argmax(intersections)
    return gt_indx, intersections[gt_indx]

def visualize(image_file, p_box,correct):
    #brutto assai...
    if os.path.exists(image_file.replace('.txt','.png')):
        image_file = image_file.replace('.txt', '.png')
    elif os.path.exists(image_file.replace('.txt','.jpg')):
        image_file = image_file.replace('.txt', '.jpg')
    else:
        print('Unsupported image type, unable to visualize {}'.format(image_file))
        return

    immy = cv2.imread(image_file)
    correct_detection = [box for cc, box in zip(correct, p_box) if cc]
    mistakes = [box for cc, box in zip(correct, p_box) if not cc]
    immy_correct = cv_show_detection.draw_prediction(immy, correct_detection)
    immy_mistake = cv_show_detection.draw_prediction(immy, mistakes)

    print('Showing correct and wrong detections for {}, press a key to continue'.format(image_file))
    cv2.imshow('correct', immy_correct)
    cv2.imshow('mistake', immy_mistake)
    cv2.waitKey()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluation of an object detection system, loads predicted bounding box and ground truth ones and compute precision and recall at different confidence threshold. The result will be saved in a csv file for further elaboration")
    parser.add_argument('-p','--predicted',help="folder containing the predicted boudning boxes",required=True)
    parser.add_argument('-l','--label',help="folder containing the ground truth bounding box",required=True)
    parser.add_argument('-o','--output',help="the result will be saved in this file in semicolon seprated format",required=True)
    parser.add_argument('-i','--image',help="folder containing the image associated with the predicted bounding boxes, used only for visualization",default=None)
    parser.add_argument('-v','--visualization',help="flag to enable visualization",action='store_true')
    parser.add_argument('--single_map', help="map gt box to one and only one detection",action='store_true')
    parser.add_argument('--iou_th',help="intersection over union thrshold for a good detction",default=0.5,type=float)
    args = parser.parse_args()

    to_check = [args.predicted,args.label,args.image] if args.visualization else [args.predicted,args.label]
    for f in to_check:
        check_existance(f)
    
    labels = sorted(glob.glob(os.path.join(args.label,'*.txt')))
    predicted = sorted(glob.glob(os.path.join(args.predicted, '*.txt')))
    assert(len(labels)==len(predicted))
    print('Found {} image to test'.format(len(labels)))

    scores = [{'TP':0,'Predicted':0}]*len(CONFIDENCE_TH)
    total_gt = 0
    for l,p in zip(labels,predicted):
        gt_boxes = read_predictions(l)
        p_boxes = read_predictions(p)
        p_boxes.sort(key=lambda x:x.confidence)
        correct = [False]*len(p_boxes)
        mapped = [{'index':None,'intersection':0}]*len(gt_boxes)
        total_gt+=len(gt_boxes)
        for index,pb in enumerate(p_boxes):
            #get the ground truth box associated with this prediction
            gt_indx,intersection = associate(pb,gt_boxes)
            gt_box = gt_boxes[gt_indx]

            #if single map and already detcted another box with bigger intersection then myself
            if args.single_map and mapped[gt_indx]['intersection'] > intersection:
                continue
            
            #compute iou and check against threshold
            iou = intersection/(pb.getArea()+gt_box.getArea())
            if iou > args.iou_th and pb.classId==gt_box.classId:
                #correct detection
                correct[index]=True

                #if single map and gt_indx is already mapped to another box fix correct array
                if args.single_map and mapped[gt_indx]['index'] is not None:
                    correct[mapped[gt_indx]['index']]=False
                mapped[gt_indx]['index']=index
                mapped[gt_indx]['intersection']=intersection

        print(correct)
        if args.visualization:
            image_file = os.path.join(args.image,os.path.basename(l))
            visualize(image_file,p_boxes,correct)

        #all prediction checked, now we need to create the precision recall curve at different confidence threshold
        for i,c in enumerate(CONFIDENCE_TH):
            id_cut=0
            for idx,p in enumerate(p_boxes):
                if p.confidence<c:
                    id_cut=idx
            scores[i]['TP']+=np.sum(correct[:id_cut])
            scores[i]['Predicted']+=id_cut
        
    #compute final values
    format_string='{};{};{};{};{}\n'
    to_write = ['IOU_TH;TP;Predictions;Precision;Recall\n']
    for c,vals in zip(CONFIDENCE_TH,scores):
        to_write.append(format_string.format(c,vlas['TP'],vals['Predicted'],vals['TP']/vals['Predicted'],vals['TP']/total_gt))
    
    print('Final Result: ')
    for l in to_write:
        print(l.replace(';','\t'))

    with open(args.output,'w+') as f_out:
        f_out.writelines(to_write)
    
    print('All Done')

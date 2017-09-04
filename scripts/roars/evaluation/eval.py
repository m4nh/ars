import os
import cv2
import argparse
import glob
import numpy as np
from roars.detections import prediction
from roars.gui import cv_show_detection
from matplotlib import pyplot as plt

CONFIDENCE_TH=[1.0-0.05*i for i in range(1,21)]

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

def visualize(image_file, p_box,correct,gt_boxes):
    #brutto assai...
    if os.path.exists(image_file.replace('.txt','.png')):
        image_file = image_file.replace('.txt', '.png')
    elif os.path.exists(image_file.replace('.txt','.jpg')):
        image_file = image_file.replace('.txt', '.jpg')
    else:
        print('Unsupported image type, unable to visualize {}'.format(image_file))
        return

    immy = cv2.imread(image_file)
    immy_copy = immy.copy()
    immy_copy_2 = immy.copy()
    correct_detection = [box for cc, box in zip(correct, p_box) if cc]
    mistakes = [box for cc, box in zip(correct, p_box) if cc==False]
    #create label_dictionary
    classes = []
    for gt in gt_boxes:
        if gt.classId not in classes:
            classes.append(gt.classId)
    cmap=cv_show_detection.getColorMap(classes)
    immy_correct = cv_show_detection.draw_prediction(immy, correct_detection,color_map=cmap,min_score_th=0.95)
    immy_mistake = cv_show_detection.draw_prediction(immy_copy, mistakes,color_map=cmap,min_score_th=0.95)
    immy_gt = cv_show_detection.draw_prediction(immy_copy_2,gt_boxes,color_map=cmap,min_score_th=0)

    print('Showing correct and wrong detections for {}, press a key to continue'.format(image_file))
    cv2.imshow('correct', immy_correct)
    cv2.imshow('mistake', immy_mistake)
    cv2.imshow('Ground Truth',immy_gt)
    cv2.waitKey()

def get_detected_for_th(gt_map,th):
    detected=0
    for g in gt_map:
        if g['max_conf']>th:
            detected+=1
    return detected

def plot(precisions,recalls,output_path=None,show=False):
    #add fake point for 0 recall
    precisions = [precisions[0]]+precisions
    recalls = [0]+recalls

    plt.figure()
    plt.title('Precision-Recall')
    plt.plot(recalls,precisions,'-o')
    axes = plt.gca()
    
    axes.set_xlim([0,1])
    axes.set_xticks(np.arange(0,1.0,0.1))
    axes.set_xlabel('Recall')

    axes.set_ylim([0,1])
    axes.set_yticks(np.arange(0,1.0,0.1))
    axes.set_ylabel('Precision')

    #save plot to file
    if output_path is not None:
        plt.savefig(output_path)
    
    if show:
        plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluation of an object detection system, loads predicted bounding box and ground truth ones and compute precision and recall at different confidence threshold. The result will be saved in a csv file for further elaboration")
    parser.add_argument('-p','--predicted',help="folder containing the predicted boudning boxes",required=True)
    parser.add_argument('-l','--label',help="folder containing the ground truth bounding box",required=True)
    parser.add_argument('-o','--output',help="Folder were the result will be saved",required=True)
    parser.add_argument('-i','--image',help="folder containing the image associated with the predicted bounding boxes, used only for visualization",default=None)
    parser.add_argument('-v','--verbosity',help="verbosity level, 0-> minimal output, 1-> show precision recall curve, >1-> show result on every single image",default=0,type=int)
    parser.add_argument('--single_map', help="map gt box to one and only one detection",action='store_true')
    parser.add_argument('--iou_th',help="intersection over union thrshold for a good detction",default=0.5,type=float)
    args = parser.parse_args()

    to_check = [args.predicted,args.label,args.image] if args.verbosity>1 else [args.predicted,args.label]
    for f in to_check:
        check_existance(f)
    
    labels = sorted(glob.glob(os.path.join(args.label,'*.txt')))
    predicted = sorted(glob.glob(os.path.join(args.predicted, '*.txt')))
    assert(len(labels)==len(predicted))
    print('Found {} image to test'.format(len(labels)))

    scores = [{'TP':0,'Predicted':0,'Detected':0} for _ in range(len(CONFIDENCE_TH))]
    total_gt = 0
    for l,p in zip(labels,predicted):
        gt_boxes = read_predictions(l)
        p_boxes = read_predictions(p)      
        p_boxes.sort(key=lambda x:x.confidence,reverse=True)


        correct = [False]*len(p_boxes)
        gt_map = [{'index':None,'intersection':0,'max_conf':0} for _ in range(len(gt_boxes))]
        total_gt+=len(gt_boxes)
        for index,pb in enumerate(p_boxes):
            #get the ground truth box associated with this prediction
            gt_indx,intersection = associate(pb,gt_boxes)
            gt_box = gt_boxes[gt_indx]

            #if single map and already detcted another box with bigger intersection then myself skip
            if args.single_map and gt_map[gt_indx]['intersection'] > intersection:
                continue
            
            #compute iou and check against threshold
            iou = intersection/(pb.getArea()+gt_box.getArea()-intersection)
            if iou > args.iou_th and pb.classId==gt_box.classId:
                #correct detection
                correct[index]=True

                #if single map and gt_indx is already gt_map to another box fix correct array
                if args.single_map and gt_map[gt_indx]['index'] is not None:
                    correct[gt_map[gt_indx]['index']]=False
                gt_map[gt_indx]['index']=index
                gt_map[gt_indx]['intersection']=intersection
                gt_map[gt_indx]['max_conf']=pb.confidence

        if args.verbosity>1:
            image_file = os.path.join(args.image,os.path.basename(l))
            visualize(image_file,p_boxes,correct,gt_boxes)

        #all prediction checked, now we need to create the precision recall curve at different confidence threshold
        for i,c in enumerate(CONFIDENCE_TH):
            id_cut=len(p_boxes)
            for idx,p in enumerate(p_boxes):
                if c>p.confidence:
                    id_cut=idx
                    break
            scores[i]['TP']+=np.sum(correct[:id_cut])
            scores[i]['Predicted']+=id_cut
            scores[i]['Detected']+=get_detected_for_th(gt_map,c)
        
    #compute final values
    format_string='{};{};{};{};{};{}\n'
    to_write = ['CONF_TH;TP;Predictions;Precision;Recall;F-Score\n']
    average_precision = 0
    last_recall = 0
    precisions=[]
    recals=[]
    for c,vals in zip(CONFIDENCE_TH,scores):
        precision=float(vals['TP'])/float(vals['Predicted'])
        precisions.append(precision)
        recall = float(vals['Detected'])/total_gt
        recals.append(recall)
        f_score = 2*((precision*recall)/(precision+recall))
        to_write.append(format_string.format(c,vals['TP'],vals['Predicted'],precision,recall,f_score))

        average_precision+=(recall-last_recall)*precision
        last_recall=recall
        
    print('Final Result: ')
    for l in to_write:
        print(l.replace(';','\t'))
    print('Average Precision: {}'.format(average_precision))

    #check if output folder exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    

    with open(os.path.join(args.output,'result.txt'),'w+') as f_out:
        f_out.writelines(to_write)
        f_out.write('\n\n Average Precision:;{}\n'.format(average_precision))
    
    plot(precisions,recals,show=args.verbosity>0,output_path=os.path.join(args.output,'precision_recall.png'))

    print('All Done')

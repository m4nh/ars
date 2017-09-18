import os
import cv2
import argparse
import glob
import numpy as np
import json
from roars.detections import prediction
from roars.gui import cv_show_detection
from roars.evaluation import utils
from matplotlib import pyplot as plt

CONFIDENCE_TH=[1.0-0.05*i for i in range(1,21)]

def visualize(image_file,correct_detection,mistakes,missed,gt_boxes):
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
    immy_copy_3 = immy.copy()
    
    #create label_dictionary
    classes = []
    for gt in gt_boxes:
        if gt.classId not in classes:
            classes.append(gt.classId)
    cmap=cv_show_detection.getColorMap(classes)
    immy_correct = cv_show_detection.draw_prediction(immy, correct_detection,color_map=cmap,min_score_th=0.01)
    immy_mistake = cv_show_detection.draw_prediction(immy_copy, mistakes,color_map=cmap,min_score_th=0.01)
    immy_gt = cv_show_detection.draw_prediction(immy_copy_2,gt_boxes,color_map=cmap,min_score_th=0)
    immy_missed = cv_show_detection.draw_prediction(immy_copy_3,missed,color_map=cmap,min_score_th=0.01)

    print('Showing correct and wrong detections for {}, press a key to continue'.format(image_file))
    cv2.imshow('Predicted Correct', immy_correct)
    cv2.imshow('Predicted Mistake', immy_mistake)
    cv2.imshow('Ground Truth',immy_gt)
    cv2.imshow('Missed',immy_missed)
    cv2.waitKey()

    

def get_detected_for_th(gt_map,th,class_id=None):
    detected=0
    for g in gt_map:
        if g['max_conf']>th and (class_id==None or g['class_id']==class_id):
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

def make_score_holder():
    return [{'TP': 0, 'Predicted': 0, 'Detected': 0, 'Total':0, 'IOU':0} for _ in range(len(CONFIDENCE_TH))]

def update_scores(prediction,correct,global_scores,class_scores,iou=None):
    class_id = prediction.classId
    conf = prediction.confidence
    for i, th in enumerate(CONFIDENCE_TH):
        if conf > th:
            if correct:
                global_scores[i]['TP'] += 1
                class_scores[class_id][i]['TP'] += 1
                global_scores[i]['IOU']+=iou
                class_scores[class_id][i]['IOU']+=iou

            global_scores[i]['Predicted'] += 1
            class_scores[class_id][i]['Predicted'] += 1

def make_table(to_write,scores):
    format_string = '{};{};{};{};{};{};{}\n'
    to_write.append('CONF_TH;TP;Predictions;Precision;Recall;F-Score;AVG_IOU;\n')
    average_precision = 0
    global_avg_IOU = 0
    last_recall = 0
    precisions = []
    recals = []
    for c, vals in zip(CONFIDENCE_TH, scores):
        precision = 0 if vals['TP'] == 0 else float(vals['TP']) / float(vals['Predicted'])
        precisions.append(precision)
        recall = 0 if vals['Total'] == 0 else float(vals['Detected']) / vals['Total']
        recals.append(recall)
        f_score = 0 if (precision == 0 and recall == 0) else 2 * ((precision * recall) / (precision + recall))
        avgIOU = 0 if vals['TP']==0 else vals['IOU']/vals['TP']
        to_write.append(format_string.format(c, vals['TP'], vals['Predicted'], precision, recall, f_score, avgIOU))

        average_precision += (recall - last_recall) * precision
        last_recall = recall

        global_avg_IOU += avgIOU

    global_avg_IOU = global_avg_IOU / len(CONFIDENCE_TH)
    to_write.append('\n\nMean Average Precision;{}\n'.format(average_precision))
    to_write.append('Mean average IOU;{}\n'.format(global_avg_IOU))

    return precisions, recals


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluation of an object detection system, loads predicted bounding box and ground truth ones and compute precision and recall at different confidence threshold. The result will be saved in a csv file for further elaboration")
    parser.add_argument('-p','--predicted',help="folder containing the predicted boudning boxes",required=True)
    parser.add_argument('-l','--label',help="folder containing the ground truth bounding box",required=True)
    parser.add_argument('-o','--output',help="output of the csv file were the result will be saved",required=True)
    parser.add_argument('-i','--image',help="folder containing the image associated with the predicted bounding boxes, used only for visualization",default=None)
    parser.add_argument('-v','--verbosity',help="verbosity level, 0-> minimal output, 1-> show precision recall curve, >1-> show result on every single image",default=0,type=int)
    parser.add_argument('--image_output',help="path were the precision recall curve will be saved, leave empty to dont save", default=None)
    parser.add_argument('--single_map', help="map gt box to one and only one detection",action='store_true')
    parser.add_argument('--iou_th',help="intersection over union thrshold for a good detction",default=0.5,type=float)
    parser.add_argument('--out_folder',help="if set to something save a json file for each image with correct mistake and missed box",default=None)
    parser.add_argument('--ignore_class',help="flag, if present ignore class predicted when computing precision and recall",action="store_true")
    args = parser.parse_args()

    if args.verbosity>1 and args.image is None:
        print('Verbosity set to max, please specify the folder were the image will be loaded with -i ${PATH}')
        exit()
    
    if args.out_folder is not None:
        if not os.path.exists(args.out_folder):
            os.makedirs(args.out_folder)

    to_check = [args.predicted,args.label,args.image] if args.verbosity>1 else [args.predicted,args.label]
    for f in to_check:
        if not utils.check_existance(f):
            exit()
    
    labels = sorted(glob.glob(os.path.join(args.label,'*.txt')))
    predicted = sorted(glob.glob(os.path.join(args.predicted, '*.txt')))
    assert(len(labels)==len(predicted))
    print('Found {} image to test'.format(len(labels)))

    global_scores = make_score_holder()
    class_scores = {}
    for img_indx,(l,p) in enumerate(zip(labels,predicted)):
        gt_boxes = utils.read_predictions(l)
        p_boxes = utils.read_predictions(p)      
        p_boxes.sort(key=lambda x:x.confidence,reverse=True)

        #create class scores for class
        for gb in gt_boxes:
            if gb.classId not in class_scores:
                class_scores[gb.classId] = make_score_holder()

        correct = [False]*len(p_boxes)
        gt_map = [{'index':None,'intersection':0,'max_conf':0,'class_id':g.classId} for g in gt_boxes]

        if len(gt_boxes)!=0:
            for index,pb in enumerate(p_boxes):
                #get the ground truth box associated with this prediction
                gt_indx,iou = utils.associate(pb,gt_boxes)
                gt_box = gt_boxes[gt_indx]

                #if single map and already detected another box with bigger iou then myself skip
                if args.single_map and gt_map[gt_indx]['intersection'] > iou:
                    continue
                
                # check against threshold
                if iou > args.iou_th and (args.ignore_class or pb.classId == gt_box.classId):
                    #correct detection
                    correct[index]=True

                    #if single map and gt_indx is already gt_map to another box fix correct array
                    if args.single_map and gt_map[gt_indx]['index'] is not None:
                        correct[gt_map[gt_indx]['index']]=False

                    gt_map[gt_indx]['index']=index
                    gt_map[gt_indx]['intersection']=iou
                    gt_map[gt_indx]['max_conf']=pb.confidence

        correct_detection = [box for cc, box in zip(correct, p_boxes) if cc]
        iou_id=[]
        iou_val=[]
        for g in gt_map:
            if g['index'] is not None:
                iou_id.append(g['index'])
                iou_val.append(g['intersection'])
        ious = [x for _, x in sorted(zip(iou_id, iou_val))]
        mistakes = [box for cc, box in zip(correct, p_boxes) if cc==False]
        missed = [box for m,box in zip(gt_map,gt_boxes) if m['index'] is None]
        if args.out_folder is not None:
            out_file = os.path.join(args.out_folder,os.path.basename(l).replace('.txt','.json'))
            with open(out_file,'w+') as f_out:
                json.dump({'correct':correct_detection,'mistakes':mistakes,'missed':missed},f_out,default=lambda o : o.__dict__)

        if args.verbosity>1:
            image_file = os.path.join(args.image,os.path.basename(l))
            visualize(image_file,correct_detection,mistakes,missed,gt_boxes)

        #all prediction checked, now we need to create the precision recall curve at different confidence threshold
        for c,ii in zip(correct_detection,ious):
            update_scores(c,True,global_scores,class_scores,ii)

        for m in mistakes:
            update_scores(m,False,global_scores,class_scores)
        
        for i,c in enumerate(CONFIDENCE_TH):
            global_scores[i]['Detected'] += get_detected_for_th(gt_map, c)
            global_scores[i]['Total']+=len(gt_boxes)
        
        for k in class_scores:
            for i,c in enumerate(CONFIDENCE_TH):
                class_scores[k][i]['Detected']+=get_detected_for_th(gt_map, c,k)
                class_scores[k][i]['Total']+=np.sum([1 for g in gt_boxes if g.classId==k])

        print('Image done: {}/{}'.format(img_indx,len(labels)),end='\r')
    
    #compute final values
    to_write = ['Global Scores\n\n']
    precisions, recals=make_table(to_write, global_scores)

    for k in class_scores:
        to_write.append('\n\nClass ID: {}\n'.format(k))
        make_table(to_write,class_scores[k])
    
    print('Final Result: ')
    for l in to_write:
        print(l.replace(';','\t\t'))
    
    parent_dir = os.path.abspath(os.path.join(args.output, os.pardir))
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    with open(args.output,'w+') as f_out:
        f_out.writelines(to_write)
    
    plot(precisions,recals,show=args.verbosity>0,output_path=args.image_output)

    print('All Done')

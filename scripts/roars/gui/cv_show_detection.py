import cv2
from roars.detections import prediction
from random import randint,seed

seed()

def random_color():
    return (randint(0,255),randint(0,255),randint(0,255))

def getColorMap(labelDictionary):
    """
    Create a dictionary that maps class_id to a random color to display it
    """
    c_map={}
    for k in labelDictionary:
        c_map[k]=random_color()
    return c_map


def draw_prediction(image,predicitons,color_map={},min_score_th=0.5,line_thickness=3,font_scale=0.7):
    """
    Draws the predictions over image and return the modified image
    Args:
        - image: image to be decorated
        - predictions: list of predictions to draw
        - color_map: dictionary of colors to be associated to each class , if None pick random colors. class_id--> key color--> value
        - min_score_th: draw detection only if it has a confidence above this th
        - line_thickeness: thickeness of the rawed rectangles
        - font_scale: dimension fo the font for write class names and scores
    Returns:
        - image: the modified image
        - color_map: the color_map used
    """
    for p in predicitons:
        #draw box only if score is above th
        if p.confidence>min_score_th:
            class_id = p.classId
            
            if class_id not in color_map:
                #color not yet setted, pick one at Random
                color_map[class_id]=random_color()
            
            current_class = p.getClassName()
            ymin, xmin, ymax, xmax = p.box()
            ymin = int(ymin*image.shape[0])
            ymax = int(ymax*image.shape[0])
            xmin = int(xmin*image.shape[1])
            xmax = int(xmax*image.shape[1])

            text_to_display = "{} - score:{:.2f}".format(current_class,p.confidence)
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color_map[class_id],line_thickness)
            cv2.putText(image,text_to_display,(xmin,ymin),cv2.FONT_HERSHEY_SIMPLEX,font_scale,255,2)
    return image
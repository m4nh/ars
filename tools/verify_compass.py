import cv2
import os
import random
import numpy as np
import glob
import sys
import argparse
import copy
import math


class Prediction(object):

    def __init__(self, row, angle_discretization, max_angle_class, image_shape, simplified=None):
        self.row = row
        self.image_shape = image_shape
        self.score = row[-1]
        self.label = int(row[0])
        self.label_expanded = int(row[0])
        self.label = int(self.label_expanded / max_angle_class)
        self.angle_class = self.label_expanded - self.label * max_angle_class

        self.angle = (self.angle_class * angle_discretization +
                      angle_discretization * 0.5) * np.pi / 180.0

        height, width, _ = self.image_shape
        self.w = row[3] * width
        self.h = row[4] * height
        self.x = row[1] * width - self.w * 0.5
        self.y = row[2] * height - self.h * 0.5
        self.center = np.array([row[1] * width, row[2] * height]).astype(int)

        self.computeRF()

    def computeRF(self):

        self.direction = np.array([np.cos(-self.angle), np.sin(self.angle)])
        self.orto_direction = np.array(
            [-np.sin(self.angle), np.cos(-self.angle)])
        self.arrow = (self.center + self.direction * self.w * 0.4).astype(int)
        self.arrow2 = (self.center + self.orto_direction *
                       self.w * 0.4).astype(int)

    def draw(self, image):
        cv2.circle(image, tuple(self.center), 3, (0, 255, 255), -1)
        cv2.line(image, tuple(self.center), tuple(self.arrow), (0, 0, 255))
        cv2.line(image, tuple(self.center), tuple(self.arrow2), (0, 255, 0))
        cv2.putText(image, "{} - {}".format(class_names[self.label], int(self.angle * 180 / np.pi)),
                    tuple(self.center), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)


class PredictionsBunch(object):

    def __init__(self):
        self.predictions = []
        self.predictions_map = {}

    def add(self, p):
        if p.label not in self.predictions_map:
            self.predictions_map[p.label] = []
        self.predictions_map[p.label] = p
        self.predictions.append(p)

    def findRadius(self, p):
        results = []
        for pred in self.predictions:
            if pred.label == p.label and pred != p:
                dist = np.linalg.norm(pred.center - p.center)
                results.append((pred, dist))
        return results


ap = argparse.ArgumentParser("Verify Compass")
ap.add_argument("--images_folder",
                default='images',
                type=str,
                help="Images Folder")
ap.add_argument("--labels_folder",
                default='labels',
                type=str,
                help="Labels Folder")
ap.add_argument("--classes",
                required=True,
                type=int,
                help="Number of classes")
ap.add_argument("--angle_discretization",
                required=True,
                type=int,
                help="Angle discretization size in Deg")
ap.add_argument("--min_score",
                required=True,
                type=float,
                help="Min score th")
ap.add_argument("--image_ext",
                default='jpg',
                type=str,
                help="Image extension")
ap.add_argument("--class_names",
                default='',
                type=str,
                help="Class names ';' separated")

args = vars(ap.parse_args())

# class names
class_names = args['class_names'].split(
    ';') if len(args['class_names']) > 0 else None
if class_names is None:
    class_names = []
    for i in range(args['classes']):
        class_names.append("Class_{}".format(i))


test_percentage = 0.1
target_folder = ''
images_folder = args['images_folder']
labels_folder = args['labels_folder']
classes_number = args['classes']
angle_discretization = args['angle_discretization']
max_angle_class = int(360.0 / angle_discretization)

labels = sorted(glob.glob(os.path.join(labels_folder, "*.txt")))
#images = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))
# for filename in os.listdir(images_folder):
#    abs_path = os.path.join(os.getcwd(), images_folder, filename)
#    images.append(abs_path)

random.shuffle(labels)

pick = labels[0]

image = os.path.join(os.path.dirname(pick), '../', images_folder,
                     os.path.splitext(os.path.basename(pick))[0] + "." + args['image_ext'])

print("Pick ", pick, image)
data = np.loadtxt(pick)
if data.ndim == 1:
    data = data.reshape(1, data.shape[0])

image = cv2.imread(image)
image_post = image.copy()
width = image.shape[1]
height = image.shape[0]

predictions_map = {}
predictions_bunch = PredictionsBunch()

print(image.shape)
for row in data:
    score = row[-1]
    if score < args['min_score']:
        continue
    prediction = Prediction(row, angle_discretization,
                            max_angle_class, image.shape)
    predictions_bunch.add(prediction)
    prediction.draw(image)


max_dist = 20.
refined = []
for pred in predictions_bunch.predictions:
    if pred.score < 0.5:
        continue
    label = pred.label
    print("Label {}".format(label), class_names[label], pred)
    nn = predictions_bunch.findRadius(pred)
    weighted_angle = 0.0
    weighted_pos = np.array([0., 0])
    counter = 0
    for pair in nn:
        if pair[1] <= max_dist:
            p = pair[0]
            angle_dist = math.atan2(
                math.sin(p.angle - pred.angle), math.cos(p.angle - pred.angle))
            weighted_angle += angle_dist * p.score
            weighted_pos += ((p.center - pred.center) * p.score)
            print(pair[0], pair[1], "SCARP!" if pair[1] > max_dist else "")
            counter += 1
    if counter > 0:
        # weighted_angle /= float(counter)
        # weighted_pos /= float(counter)
        clone = copy.deepcopy([pred])[0]
        clone.angle += weighted_angle
        clone.center += weighted_pos.astype(int)
        clone.computeRF()
    else:
        clone = pred
    refined.append(clone)
# for label, pred in predictions_map.items():

#     weighted_angle = 0.0
#     weighted_pos = np.array([0., 0])
#     for p in pred:
#         weighted_pos += (p.center * p.score)
#         weighted_angle += p.score * p.angle
#         print(p.score)
#     weighted_angle /= float(len(pred))
#     weighted_pos /= float(len(pred))
#     print("W: ", weighted_angle, weighted_pos)
    # clone = copy.deepcopy([pred[0]])[0]
    # clone.angle = weighted_angle
    # clone.center = weighted_pos.astype(int)
    # clone.computeRF()
#     print(pred[0].angle)
#     print(clone.angle)
#     refined.append(clone)

for r in refined:
    r.draw(image_post)


cv2.namedWindow("image", cv2.WINDOW_FULLSCREEN)
cv2.imshow("image", image)
cv2.imshow("image2", image_post)
cv2.waitKey(0)

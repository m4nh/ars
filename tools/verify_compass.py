import cv2
import os
import random
import numpy as np
import glob
import sys
import argparse


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
args = vars(ap.parse_args())


test_percentage = 0.1
target_folder = ''
images_folder = args['images_folder']
labels_folder = args['labels_folder']
classes_number = args['classes']
angle_discretization = args['angle_discretization']
max_angle_class = int(360.0 / angle_discretization)

images = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))
# for filename in os.listdir(images_folder):
#    abs_path = os.path.join(os.getcwd(), images_folder, filename)
#    images.append(abs_path)

random.shuffle(images)

pick = images[0]

print(pick)
path, file = os.path.split(pick)
path, folder = os.path.split(path)


file = file.split(".")[0] + ".txt"
rebuilt_path = os.path.join(path, labels_folder, file)


data = np.loadtxt(rebuilt_path)
if data.ndim == 1:
    data = data.reshape(1, data.shape[0])

image = cv2.imread(pick)
width = image.shape[1]
height = image.shape[0]

print(image.shape)
for row in data:
    #row = row.reshape(5, 1)
    label_expanded = int(row[0])
    label = int(label_expanded / max_angle_class)
    angle_class = label_expanded - label * max_angle_class
    angle = angle_class * angle_discretization * np.pi / 180.0

    w = row[3] * width
    h = row[4] * height
    x = row[1] * width - w * 0.5
    y = row[2] * height - h * 0.5

    center = np.array([row[1] * width, row[2] * height]).astype(int)
    direction = np.array([np.cos(-angle), np.sin(angle)])
    arrow = (center + direction * w * 0.8).astype(int)

    print(x, y)
    cv2.rectangle(image, (int(x), int(y)), (int(
        x + w), int(y + h)), (255, 0, 255), 2)
    cv2.circle(image, tuple(center), 3, (0, 255, 255), -1)
    cv2.line(image, tuple(center), tuple(arrow), (0, 255, 0))
    cv2.putText(image, "C{}".format(label + 1),
                (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("image", image)
cv2.waitKey(0)

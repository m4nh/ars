import cv2
import os
import random
import numpy as np
import glob
import sys

test_percentage = 0.1
target_folder = ''
images_folder = 'images'


images = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))
# for filename in os.listdir(images_folder):
#    abs_path = os.path.join(os.getcwd(), images_folder, filename)
#    images.append(abs_path)

random.shuffle(images)

pick = images[0]

print(pick)
path, file = os.path.split(pick)
path, folder = os.path.split(path)

folder = sys.argv[1]
file = file.split(".")[0] + ".txt"
rebuilt_path = os.path.join(path, folder, file)


data = np.loadtxt(rebuilt_path)
if data.ndim == 1:
    data = data.reshape(1, data.shape[0])

image = cv2.imread(pick)
width = image.shape[1]
height = image.shape[0]

print(image.shape)
for row in data:
    #row = row.reshape(5, 1)
    label = int(row[0])

    w = row[3] * width
    h = row[4] * height
    x = row[1] * width - w * 0.5
    y = row[2] * height - h * 0.5
    print(x, y)
    cv2.rectangle(image, (int(x), int(y)), (int(
        x + w), int(y + h)), (255, 0, 255), 2)
    cv2.putText(image, "C{}".format(label + 1),
                (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow("image", image)
cv2.waitKey(0)

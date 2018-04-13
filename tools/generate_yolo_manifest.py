import os
import glob
import sys
import random

classes = sys.argv[2]
images = sys.argv[1]
images = os.path.abspath(images)
labels = os.path.abspath(os.path.join(images, "../labels"))
output_folder = os.path.abspath(os.path.join(images, "../"))

images = sorted(glob.glob(os.path.join(images, "*.*")))
labels = sorted(glob.glob(os.path.join(labels, "*.*")))


val_perc = 0.2

z = images
random.shuffle(z)

val_size = int(len(z) * val_perc)

val_list = z[0:val_size]
train_list = z[val_size:]

fv = open(os.path.join(output_folder, 'validationlist.txt'), 'w')
ft = open(os.path.join(output_folder, 'traininglist.txt'), 'w')
fc = open(os.path.join(output_folder, 'className.txt'), 'w')

for item in val_list:
    fv.write(item)
    fv.write("\n")
for item in train_list:
    ft.write(item)
    ft.write("\n")

for i in range(int(classes)):
    fc.write('Class_{}'.format(i))
    fc.write("\n")

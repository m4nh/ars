from dicttoxml import dicttoxml
from xml.dom import minidom

import xml.etree.ElementTree as ET
from xml.etree import ElementTree
import sys
import os
import glob
import numpy as np
import cv2

classes = None  # Map label(int)-> string

folder = sys.argv[1]
output_folder = sys.argv[2]

try:
    os.mkdir(output_folder)
except:
    pass


image_folder = os.path.join(folder, 'images')
label_folder = os.path.join(folder, 'labels')


images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
labels = sorted(glob.glob(os.path.join(label_folder, "*.txt")))


for index in range(0, len(images)):
    print(float(index) / float(len(images)) * 100)
    image = images[index]
    label = np.loadtxt(labels[index])
    img = cv2.imread(image)
    # print image
    # print label
    # print img.shape
    w = float(img.shape[1])
    h = float(img.shape[0])
    objects = []
    for r in label:
        if classes is not None:
            cls = classes[int(r[0])]
        else:
            cls = "Class_{}".format(int(r[0]))

        ow = int(r[3] * w)
        oh = int(r[4] * h)
        ox = int(r[1] * w - ow * 0.5)
        oy = int(r[2] * h - oh * 0.5)
        angle = 0.0
        if len(r) >= 5:
            angle = float(r[5])

        obj = {
            'name': cls,
            'pose': angle,
            'truncated': 0,
            'difficult': 0,
            'bndbox': {
                'xmin': ox,
                'ymin': oy,
                'xmax': ox + ow,
                'ymax': oy + oh
            }
        }
        objects.append(obj)
        # print cls, ox, oy, ow, oh

    data = {
        'folder': 'images',
        'filename': os.path.basename(image),
        'path': image,
        'source': {'database': 'Unknown'},
        'size': {
            'width': int(w),
            'height': int(h),
            'depth': img.shape[2]
        }
    }

    xmlstr = dicttoxml(data, custom_root='annotation', attr_type=False)
    document = ET.fromstring(xmlstr)

    for i, o in enumerate(objects):
        #print(i, o)
        ostr = dicttoxml(o, attr_type=False, custom_root='object').replace(
            '<?xml version="1.0" encoding="UTF-8" ?>', '')
        ostr = dicttoxml(o, attr_type=False, custom_root='object').replace(
            '<?xml version="1.0" ?>', '')

        oel = ET.fromstring(ostr)
        document.append(oel)

    xml = minidom.parseString(ElementTree.tostring(document).replace(
        '<?xml version="1.0" ?>', '').replace(
        '<?xml version="1.0" encoding="UTF-8" ?>', ''))
    # print(xml.toprettyxml())

    fakedoc = minidom.Document()
    declaration = fakedoc.toxml()

    out_file = os.path.join(output_folder, os.path.splitext(
        os.path.basename(image))[0] + ".xml")
    f = open(out_file, 'w')
    f.write(xml.toprettyxml()[len(declaration):].strip())
    f.close()

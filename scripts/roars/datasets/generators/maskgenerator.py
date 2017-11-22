#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.datasets.datasetutils import DatasetBuilder, TrainingClass
import os
import numpy as np
import cv2


class MaskDatasetBuilder(DatasetBuilder):
    ZERO_PADDING_SIZE = 5

    def __init__(self, training_dataset, dest_folder, jumps=5, val_percentage=0.05, test_percentage=0.1, randomize_frames=False):
        super(MaskDatasetBuilder, self).__init__(training_dataset, dest_folder)
        self.jumps = jumps
        self.val_percentage = val_percentage
        self.test_percentage = test_percentage
        self.randomize_frames = randomize_frames

    def build(self, options={}):

        if os.path.exists(self.dest_folder) or len(self.dest_folder) == 0:
            return False

        os.mkdir(self.dest_folder)

        frames = self.training_dataset.getAllFrames()

        counter = 0
        for frame in frames:
            if counter % self.jumps == 0:
                counter_string = '{}'.format(str(int(counter / self.jumps)).zfill(
                    MaskDatasetBuilder.ZERO_PADDING_SIZE))

                gts = frame.getInstancesBoxesWithLabels()

                img = cv2.imread(frame.getImagePath())
                pair = np.ones(img.shape, dtype=np.uint8) * 255
                for inst in gts:
                    hull = cv2.convexHull(np.array(inst[1]))
                    cv2.fillConvexPoly(
                        pair, hull, TrainingClass.getColorByLabel(inst[0]))

                whole = np.hstack((pair, img))

                img_file = os.path.join(
                    self.dest_folder, counter_string + ".jpg")
                print "Writing to", img_file
                cv2.imwrite(img_file, whole)

            counter = counter + 1

        return True

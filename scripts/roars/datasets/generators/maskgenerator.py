#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.datasets.datasetutils import DatasetBuilder, TrainingClass
import os
import numpy as np
import cv2
import shutil
import roars.geometry.transformations as transformations


class MaskDatasetBuilder(DatasetBuilder):
    ZERO_PADDING_SIZE = 5

    def __init__(self, training_dataset, dest_folder, jumps=5, val_percentage=0.05, test_percentage=0.1, randomize_frames=False, boxed_instances=False):
        super(MaskDatasetBuilder, self).__init__(
            training_dataset, dest_folder)
        self.jumps = jumps
        self.val_percentage = val_percentage
        self.test_percentage = test_percentage
        self.randomize_frames = randomize_frames
        self.boxed_instances = boxed_instances

    def build(self, options={}):

        if os.path.exists(self.dest_folder) or len(self.dest_folder) == 0:
            return False

        os.mkdir(
            self.dest_folder)

        frames = self.training_dataset.getAllFrames()

        counter = 0
        for frame in frames:
            if counter % self.jumps == 0:
                counter_string = '{}'.format(str(int(counter / self.jumps)).zfill(MaskDatasetBuilder.ZERO_PADDING_SIZE))

                img = cv2.imread(frame.getImagePath())

                #⬢⬢⬢⬢⬢➤ Compute Unique Labels
                instances = frame.getInstances()
                labels = []
                for inst in instances:
                    labels.append(inst.label)
                labels = set(labels)

                #⬢⬢⬢⬢⬢➤ Masks for each label
                for l in labels:

                    gts = frame.getInstancesBoxesWithLabels(filter_labels=[l])
                    pair = np.ones(img.shape, dtype=np.uint8) * 255
                    for inst in gts:
                        if not self.boxed_instances:
                            hull = cv2.convexHull(np.array(inst[1]))
                            cv2.fillConvexPoly(pair, hull, TrainingClass.getColorByLabel(inst[0]))
                        else:
                            hull = cv2.boundingRect(np.array(inst[1]))
                            cv2.rectangle(pair, (hull[0], hull[1]), (hull[0] + hull[2], hull[1] + hull[3]), TrainingClass.getColorByLabel(inst[0]), -1)
                    mask_img_file = os.path.join(self.dest_folder, counter_string + "_mask_{}.png".format(l))
                    cv2.imwrite(mask_img_file, pair)

                #⬢⬢⬢⬢⬢➤ Mask for all labels together
                gts = frame.getInstancesBoxesWithLabels()
                pair = np.ones(img.shape, dtype=np.uint8) * 255
                for inst in gts:
                    if not self.boxed_instances:
                        hull = cv2.convexHull(np.array(inst[1]))
                        cv2.fillConvexPoly(pair, hull, TrainingClass.getColorByLabel(inst[0]))
                    else:
                        hull = cv2.boundingRect(np.array(inst[1]))
                        cv2.rectangle(pair, (hull[0], hull[1]), (hull[0] + hull[2], hull[1] + hull[3]), TrainingClass.getColorByLabel(inst[0]), -1)
                mask_img_file = os.path.join(self.dest_folder, counter_string + "_mask_all.png")
                cv2.imwrite(mask_img_file, pair)

                #⬢⬢⬢⬢⬢➤ RGB Image
                rgb_img_file = os.path.join(self.dest_folder, counter_string + "_rgb.jpg")
                cv2.imwrite(rgb_img_file, img)
                print "Writing to", rgb_img_file

                #⬢⬢⬢⬢⬢➤ Depth Image
                if frame.getImageDepthPath():
                    depth_img_file = frame.getImageDepthPath()
                    ext = os.path.splitext(depth_img_file)[1]
                    depth_img_file = os.path.join(self.dest_folder, counter_string + "_depth." + ext)
                    shutil.copy(frame.getImageDepthPath(), depth_img_file)

                #⬢⬢⬢⬢⬢➤ Camera Pose
                camera_pose = frame.getCameraPose()
                camera_pose = transformations.KLDtoNumpyMatrix(camera_pose)
                pose_file = os.path.join(self.dest_folder, counter_string + "_camerapose.txt")
                np.savetxt(pose_file, camera_pose)

            counter = counter + 1

        return True

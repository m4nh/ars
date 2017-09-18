#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import os
import PyKDL
import cv2
import math
import sys

import roars.geometry.transformations as transformations
from roars.geometry.planes import PlaneFrom4Points

###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################


class VirtualObjectGroundTruth(object):

    def __init__(self, path, dataset_name, delimiter=' '):
        self.path = path
        self.dataset_name = dataset_name
        self.delimiter = delimiter
        self._loadFromFile()
        self._loadCamera()

    def getGTName(self):
        return os.path.join(self.path, self.dataset_name + "_gt.txt")

    def getCameraName(self):
        return os.path.join(self.path, "camera.txt")

    def _loadCamera(self):
        cam = np.loadtxt(self.getCameraName())
        quaternion = cam[3:7]
        quaternion = quaternion / np.linalg.norm(quaternion)
        self.camera_frame = PyKDL.Frame(
            PyKDL.Rotation.Quaternion(
                quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
            PyKDL.Vector(
                cam[0], cam[1], cam[2]
            )
        )

    def _loadFromFile(self):
        self.data = np.genfromtxt(
            self.getGTName(),
            delimiter=self.delimiter,
            dtype=None,
            names=True)
        self.objs = {}
        for row in self.data:
            name = row[0]
            if 'fake' in name:
                continue
            size = [row[1], row[2], row[3]]
            p = PyKDL.Vector(row[4], row[5], row[6])
            rot = PyKDL.Rotation.Quaternion(
                row[7],
                row[8],
                row[9],
                row[10]
            )
            color = np.array([row[11], row[12], row[13]])

            self.objs[name] = {
                "vobj": VirtualObject(
                    frame=PyKDL.Frame(rot, p),
                    size=size
                ),
                "color": color
            }

    def getVirtualObjects(self):
        return self.objs


###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

class VirtualObjectOptions(object):

    def __init__(self):
        self.translate_z = True


class VirtualObject(PyKDL.Frame):
    DEFAULT_PREFIX_NAME = "VO"

    def __init__(self, frame=PyKDL.Frame(), size=np.array([0.1, 0.1, 0.1]), label=""):
        super(VirtualObject, self).__init__()
        self.p = frame.p
        self.M = frame.M
        self.size = size
        self.label = label

    def getName(self):
        return VirtualObject.DEFAULT_PREFIX_NAME + "_{}".format(self.label)

    def getObjectPoints(self, options=VirtualObjectOptions(), camera_frame=PyKDL.Frame()):

        p0 = self.p
        sx = self.size[0]
        sy = self.size[1]
        sz = self.size[2]

        if options.translate_z:
            trans_z = PyKDL.Vector(0, 0, sz * 0.5)
        else:
            trans_z = PyKDL.Vector(0, 0, 0)

        box_mat = np.array(
            [
                [+1, +1, -1, -1, +1, +1, -1, -1],
                [-1, +1, +1, -1, -1, +1, +1, -1],
                [+1, +1, +1, +1, -1, -1, -1, -1]
            ]
        )
        box_mat = 0.5 * box_mat
        box_mat = np.transpose(box_mat)

        size_mat = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])
        points = np.matmul(box_mat, size_mat)

        out_points = np.zeros((8, 3, 1), dtype="float32")
        row_i = 0
        for row in points:
            p = PyKDL.Vector(row[0], row[1], row[2]) + trans_z
            out_points[row_i, 0] = p.x()
            out_points[row_i, 1] = p.y()
            out_points[row_i, 2] = p.z()
            row_i += 1
        return out_points

    def getImagePoints(self, options=VirtualObjectOptions(), camera_frame=PyKDL.Frame(), camera=None):
        if camera == None:
            return []
        obj_points = self.getObjectPoints(
            camera_frame=camera_frame,
            options=options
        )
        object_to_camera = camera_frame.Inverse() * self
        cRvec, cTvec = transformations.KDLToCv(object_to_camera)
        img_points, _ = cv2.projectPoints(
            obj_points,
            cRvec,
            cTvec,
            np.array(camera.camera_matrix),
            np.array(camera.distortion_coefficients)
        )
        points = []
        for i in range(0, len(img_points)):
            points.append((int(img_points.item(i, 0, 0)),
                           int(img_points.item(i, 0, 1))))
        return points

    def getImageFrame(self, points=None,  options=VirtualObjectOptions(), grow_factor=1.0, camera_frame=PyKDL.Frame(), camera=None, only_top_face=True):
        if points == None:
            points = self.getImagePoints(
                options=options,
                camera=camera,
                camera_frame=camera_frame
            )

        xs = []
        ys = []
        if only_top_face:
            for i in range(0, 4):
                p = points[i]
                xs.append(p[0])
                ys.append(p[1])
        else:
            for p in points:
                xs.append(p[0])
                ys.append(p[1])

        max_x = max(xs)
        max_y = max(ys)
        min_x = min(xs)
        min_y = min(ys)

        max_x = max(max_x, 0)
        max_y = max(max_y, 0)
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)

        max_x = min(max_x, camera.width)
        min_x = max(min_x, 0)
        max_y = min(max_y, camera.height)
        min_y = max(min_y, 0)

        p0 = np.array([min_x, min_y])
        p1 = np.array([max_x, min_y])
        p2 = np.array([max_x, max_y])
        p3 = np.array([min_x, max_y])

        w = math.fabs(max_x - min_x) / camera.width
        h = math.fabs(max_y - min_y) / camera.height
        x = float(p0[0]) / float(camera.width)
        y = float(p0[1]) / float(camera.height)
        return VirtualObject.enlargeFrame([x, y, w, h], 1.0, True)

    def draw(self, img, camera_frame=PyKDL.Frame(), camera=None, color=np.array([255, 0, 255]), thickness=2):
        img_points = self.getImagePoints(
            camera_frame=camera_frame,
            camera=camera
        )
        VirtualObject.drawBox(img_points, img, color, thickness)

    @staticmethod
    def drawBox(points, output, color=np.array([255, 0, 255]), thickness=2):
        try:
            cv2.line(output, points[0], points[1], color, thickness)
            cv2.line(output, points[1], points[2], color, thickness)
            cv2.line(output, points[2], points[3], color, thickness)
            cv2.line(output, points[3], points[0], color, thickness)

            cv2.line(output, points[4], points[5], color, thickness)
            cv2.line(output, points[5], points[6], color, thickness)
            cv2.line(output, points[6], points[7], color, thickness)
            cv2.line(output, points[7], points[4], color, thickness)

            cv2.line(output, points[0], points[4], color, thickness)
            cv2.line(output, points[1], points[5], color, thickness)
            cv2.line(output, points[2], points[6], color, thickness)
            cv2.line(output, points[3], points[7], color, thickness)
        except:
            pass

    @staticmethod
    def drawFrame(frame_data, output, color=np.array([255, 0, 255]), thickness=2):

        width = output.shape[1]
        height = output.shape[0]

        try:
            x = int(frame_data[0] * width -
                    frame_data[2] * width * 0.5)
            y = int(frame_data[1] * height -
                    frame_data[3] * height * 0.5)
            w = int(frame_data[2] * width)
            h = int(frame_data[3] * height)

            cv2.line(output, (x, y), (x + w, y), color, thickness)
            cv2.line(output, (x + w, y), (x + w, y + h), color, thickness)
            cv2.line(output, (x + w, y + h), (x, y + h), color, thickness)
            cv2.line(output, (x, y + h), (x, y), color, thickness)
        except:
            pass

    @staticmethod
    def isValidFrame(frame_data):
        if len(frame_data) == 4:
            if frame_data[0] > 0:
                if frame_data[1] > 0:
                    if frame_data[2] > 0:
                        if frame_data[3] > 0:
                            return True
        return False

    @staticmethod
    def enlargeFrame(frame_data, grow_factor, center_coordinates=False):
        x = frame_data[0]
        y = frame_data[1]
        w = frame_data[2]
        h = frame_data[3]
        nw = w * grow_factor
        nh = h * grow_factor
        wdiff = (nw - w) * 0.5
        hdiff = (nh - h) * 0.5
        nx = x - wdiff
        ny = y - hdiff
        if center_coordinates:
            nx = nx + w * 0.5
            ny = ny + h * 0.5
        return [nx, ny, nw, nh]

    # def __mul__(self, other):
    #     f1 = PyKDL.Frame()
    #     f1.M = self.M
    #     f1.p = self.p
    #     f2 = PyKDL.Frame()
    #     f2.M = other.M
    #     f2.p = other.p
    #     f3 = f1 * f2
    #     return VirtualObject(frame=f3, size=self.size, label=self.label)

    # def __rmul__(self, other):
    #     f1 = PyKDL.Frame()
    #     f1.M = self.M
    #     f1.p = self.p
    #     f2 = PyKDL.Frame()
    #     f2.M = other.M
    #     f2.p = other.p
    #     f3 = f2 * f1
    #     return VirtualObject(frame=f3, size=self.size, label=self.label)


###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

class GenericBoundingBox(object):

    def __init__(self):
        self.rf = None
        self.size = np.array([0, 0, 0])
        self.valid = False

    def buildVirtualObject(self):
        return VirtualObject(self.rf, self.size)

    def isValid(self):
        return self.valid

    def getSize(self):
        return self.size

    def getRF(self):
        return self.rf

###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################


class BoundingBoxFromSixPoints(GenericBoundingBox):

    def __init__(self, points):
        super(BoundingBoxFromSixPoints, self).__init__()
        self.plane_rf = None
        self.valid = self.buildFromPoints(points)

    def buildFromPoints(self, points):
        if len(points) == 6:
            mat = self.pointsMatrix(points)
            plane_subset = points[0:4]
            plane = PlaneFrom4Points(self.pointsMatrix(plane_subset).T)
            if plane.isValid():
                self.plane_rf = plane.rf

                #⬢⬢⬢⬢⬢➤ Z SIZE relative to Plane
                p5 = points[4]
                p6 = points[5]
                p5 = plane.getRelativePoint(p5)
                p6 = plane.getRelativePoint(p6)
                delta_z, z_size = self._computeZBound(p5, p6)
                trans_z = PyKDL.Frame(
                    PyKDL.Vector(
                        0,
                        0,
                        delta_z - 0.5 * z_size
                    )
                )
                self.rf = self.plane_rf * trans_z

                self.size = np.array([
                    plane.size[0],
                    plane.size[1],
                    z_size
                ])
                return True
        return False

    def _computeZBound(self, p1, p2):
        t_z = p1[2]
        b_z = p2[2]
        if b_z > t_z:
            t_z, b_z = b_z, t_z
        return t_z, math.fabs(t_z - b_z)

    def pointsMatrix(self, points):
        mat = np.zeros((len(points), 3))
        for r in range(0, len(points)):
            mat[r, :] = points[r]
        return mat


###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

class BoundingBoxFromFourPoints(GenericBoundingBox):

    def __init__(self, points):
        super(BoundingBoxFromFourPoints, self).__init__()
        self.valid = self.buildFromPoints(points)

    def buildFromPoints(self, points):
        if len(points) == 4:
            p0 = points[0]
            p1 = points[1]
            p2 = points[2]
            p3 = points[3]

            vx = (p1 - p0) / np.linalg.norm((p1 - p0))
            vz = (p2 - p1) / np.linalg.norm((p2 - p1))
            vy = np.cross(vz, vx)

            s = np.array([
                np.linalg.norm(p1 - p0),
                np.linalg.norm(p3 - p2),
                np.linalg.norm(p2 - p1)
            ])

            frame = PyKDL.Frame()
            frame.M = PyKDL.Rotation(
                vx[0], vy[0], vz[0],
                vx[1], vy[1], vz[1],
                vx[2], vy[2], vz[2]
            )

            frame.p = PyKDL.Vector(
                p0[0],
                p0[1],
                p0[2]
            )

            delta = PyKDL.Frame(PyKDL.Vector(
                s[0] * 0.5, s[1] * 0.5, 0.0
            ))
            frame = frame * delta

            self.rf = frame
            self.size = s
            return True
        return False


###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################


class BoundingBoxGenerator(object):

    @staticmethod
    def getGenerator(number_of_points):
        if number_of_points == 6:
            return BoundingBoxFromSixPoints
        if number_of_points == 4:
            return BoundingBoxFromFourPoints
        else:
            return None

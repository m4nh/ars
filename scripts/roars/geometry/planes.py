import numpy as np
import PyKDL
import roars.geometry.transformations as transformations
import math
import sys

###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################


class PlaneFrom4Points(object):

    def __init__(self, points):
        self.points = points.T
        self.proj_points = []
        self.relative_points = []
        self.height_points = []
        self.box = None
        self.box_rf = None
        self.size = [0, 0, 0]
        self.rf = None
        self.plane_data = self.planeFit(points)
        if self.plane_data:
            self.normal = self.plane_data[1]
            self.centroid = self.plane_data[0]
            self.a = self.normal[0]
            self.b = self.normal[1]
            self.c = self.normal[2]
            self.d = - \
                (self.a * self.centroid[0] + self.b *
                 self.centroid[1] + self.c * self.centroid[2])

            self.buildRF()

    def buildRF(self):

        self.proj_points = []
        for i in range(0, 4):
            self.proj_points.append(
                self.getPlanePoint(self.points[i][0], self.points[i][1])
            )

        vx = self.proj_points[1] - self.proj_points[0]
        vx = vx / np.linalg.norm(vx)

        vy = np.cross(self.normal, vx)
        vz = self.normal
        rf = np.array([
            [vx[0], vy[0], vz[0], self.centroid[0]],
            [vx[1], vy[1], vz[1], self.centroid[1]],
            [vx[2], vy[2], vz[2], self.centroid[2]],
            [0, 0, 0, 1]
        ])

        self.rf = transformations.NumpyMatrixToKDL(rf)
        self.inv_rf = np.linalg.inv(rf)

        self.relative_points = []
        for p in self.proj_points:
            ep = np.array([p[0], p[1], p[2], 1])
            ep = np.matmul(self.inv_rf, ep)
            self.relative_points.append(ep)

        min_x = sys.float_info.max
        min_y = sys.float_info.max
        max_x = sys.float_info.min
        max_y = sys.float_info.min

        for pp in self.relative_points:
            if pp[0] > max_x:
                max_x = pp[0]
            if pp[0] < min_x:
                min_x = pp[0]
            if pp[1] > max_y:
                max_y = pp[1]
            if pp[1] < min_y:
                min_y = pp[1]

        self.size[0] = math.fabs(max_x - min_x)
        self.size[1] = math.fabs(max_y - min_y)

        return self.rf

    def getRelativePoint(self, p):
        ep = np.array([p[0], p[1], p[2], 1])
        ep = np.matmul(self.inv_rf, ep)
        return np.array([ep[0], ep[1], ep[2]])

    def getPlanePoint(self, x, y):
        z = (self.a * x + self.b * y + self.d) / (- self.c)
        return np.array([x, y, z])

    def isValid(self):
        return self.rf != None

    def planeFit(self, points):
        points = np.reshape(points, (np.shape(points)[0], -1))
        if points.shape[0] > points.shape[1]:
            return None
        ctr = points.mean(axis=1)
        x = points - ctr[:, np.newaxis]
        M = np.dot(x, x.T)
        return ctr, np.linalg.svd(M)[0][:, -1]

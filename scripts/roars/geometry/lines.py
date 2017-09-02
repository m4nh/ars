import numpy as np
import PyKDL


def lineLineIntersection(rays):

    lines = []
    points = []
    for r in rays:
        l = np.array([
            r[1][0],
            r[1][1],
            r[1][2]
        ]).reshape(3, 1)

        p = np.array([
            r[0][0],
            r[0][1],
            r[0][2]
        ]).reshape(3, 1)
        l = l / np.linalg.norm(l)

        lines.append(l)
        points.append(p)

    v_l = np.zeros((3, 3))
    v_r = np.zeros((3, 1))
    for index in range(0, len(lines)):
        l = lines[index]
        p = points[index]
        v = np.eye(3) - np.matmul(l, l.T)
        v_l = v_l + v
        v_r = v_r + np.matmul(v, p)

    x = np.matmul(np.linalg.pinv(v_l), v_r)
    return x


def compute3DRay(point_2d, camera_matrix_inv, camera_pose):
    '''
    Computes 3D Ray from camera
    '''

    point_2d = np.array([
        point_2d[0],
        point_2d[1],
        1.0
    ]).reshape(3, 1)
    ray = np.matmul(camera_matrix_inv, point_2d)
    ray = ray / np.linalg.norm(ray)
    ray = ray.reshape(3)

    ray_v = PyKDL.Vector(ray[0], ray[1], ray[2])
    ray_v = camera_pose * ray_v
    center = camera_pose.p
    ray_dir = ray_v - center
    ray_dir.Normalize()

    line = (
        np.array([center[0], center[1], center[2]]),
        np.array([ray_dir[0], ray_dir[1], ray_dir[2]])
    )
    return line

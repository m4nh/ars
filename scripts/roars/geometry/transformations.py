import math
import numpy
import PyKDL
import cv2

# epsilon for testing whether a number is close to zero
_EPS = numpy.finfo(float).eps * 4.0


def selfProjection(frame):
    if isinstance(frame, PyKDL.Frame):
        newframe = PyKDL.Frame()
        newframe.p.x(frame.p.x())
        newframe.p.y(frame.p.y())
        newframe.p.z(0)
        return newframe
    return frame


def tfToKDL(tf):
    frame = PyKDL.Frame()
    frame.p.x(tf[0][0])
    frame.p.y(tf[0][1])
    frame.p.z(tf[0][2])

    frame.M = PyKDL.Rotation.Quaternion(
        tf[1][0],
        tf[1][1],
        tf[1][2],
        tf[1][3]
    )
    return frame


def PoseToKDL(pose):
    frame = PyKDL.Frame()
    frame.p = PyKDL.Vector(
        pose.position.x,
        pose.position.y,
        pose.position.z
    )
    frame.M = PyKDL.Rotation.Quaternion(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    )
    return frame


def KDLtoTf(frame):
    tf = [
        [frame.p.x(), frame.p.y(), frame.p.z()],
        frame.M.GetQuaternion()
    ]
    return tf


def cvToKDL(Rvec, Tvec):
    """ Converts the OpenCV couple to PyKDL Frame """
    rot, _ = cv2.Rodrigues(Rvec)
    frame = PyKDL.Frame()
    frame.M = PyKDL.Rotation(
        rot[0, 0], rot[0, 1], rot[0, 2],
        rot[1, 0], rot[1, 1], rot[1, 2],
        rot[2, 0], rot[2, 1], rot[2, 2]
    )
    frame.p = PyKDL.Vector(
        Tvec[0],
        Tvec[1],
        Tvec[2]
    )
    return frame


def KDLtoNumpyVector(frame, fmt='XYZQ'):
    if fmt == 'XYZQ':
        p = frame.p
        q = frame.M.GetQuaternion()
        return numpy.array([
            p.x(), p.y(), p.z(), q[0], q[1], q[2], q[3]
        ]).reshape(1, 7)
    elif fmt == 'RPY':
        p = frame.p
        roll, pitch, yaw = frame.M.GetRPY()
        return numpy.array([
            p.x(), p.y(), p.z(), roll, pitch, yaw
        ]).reshape(1, 6)


def KLDtoNumpyMatrix(frame):
    M = frame.M
    R = numpy.array([
        [M[0, 0], M[0, 1], M[0, 2]],
        [M[1, 0], M[1, 1], M[1, 2]],
        [M[2, 0], M[2, 1], M[2, 2]],
    ])
    P = numpy.transpose(
        numpy.array([
            frame.p.x(),
            frame.p.y(),
            frame.p.z()
        ])
    )
    P = P.reshape(3, 1)
    T = numpy.concatenate([R, P], 1)
    T = numpy.concatenate([T, numpy.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    return T


def NumpyMatrixToKDL(matrix):
    frame = PyKDL.Frame()
    for i in range(0, 3):
        for j in range(0, 3):
            frame.M[i, j] = matrix[i, j]
    frame.p = PyKDL.Vector(
        matrix[0, 3],
        matrix[1, 3],
        matrix[2, 3]
    )
    return frame


def buildProjectionMatrix(camera_pose, camera_matrix):
    T = camera_pose.Inverse()
    T_sub = KLDtoNumpyMatrix(T)[0:3, :]
    proj_matrix = numpy.matmul(camera_matrix, T_sub)
    return proj_matrix


def NumpyVectorToKDLVector(array):
    return PyKDL.Vector(
        array[0],
        array[1],
        array[2]
    )


def NumpyVectorToKDL(array):
    frame = PyKDL.Frame()
    frame.p = PyKDL.Vector(
        array[0],
        array[1],
        array[2]
    )
    quaternion = array[3:7]
    quaternion = quaternion / numpy.linalg.norm(quaternion)
    frame.M = PyKDL.Rotation.Quaternion(
        quaternion[0],
        quaternion[1],
        quaternion[2],
        quaternion[3]
    )
    return frame


def FrameVectorFromKDL(frame):
    frame_q = frame.M.GetQuaternion()
    frame_v = [frame.p[0],
               frame.p[1],
               frame.p[2],
               frame_q[0],
               frame_q[1],
               frame_q[2],
               frame_q[3], ]
    return frame_v


def FrameVectorToKDL(frame_v):
    frame = PyKDL.Frame(PyKDL.Vector(frame_v[0], frame_v[1], frame_v[2]))
    q = numpy.array([frame_v[3], frame_v[4], frame_v[5], frame_v[6]])
    q = q / numpy.linalg.norm(q)
    frame.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
    return frame


def ListToKDLVector(list):
    return PyKDL.Vector(list[0], list[1], list[2])


def KDLVectorToList(vec):
    return [vec.x(), vec.y(), vec.z()]


def KDLFromArray(chunks, fmt='RPY'):
    if fmt == 'RPY':
        frame = PyKDL.Frame()
        frame.p = PyKDL.Vector(
            chunks[0], chunks[1], chunks[2]
        )
        frame.M = PyKDL.Rotation.RPY(
            chunks[3],
            chunks[4],
            chunks[5]
        )
    if fmt == 'XYZQ':
        frame = PyKDL.Frame()
        frame.p = PyKDL.Vector(
            chunks[0], chunks[1], chunks[2]
        )
        q = numpy.array([chunks[3],
                         chunks[4],
                         chunks[5],
                         chunks[6]])
        q = q / numpy.linalg.norm(q)
        frame.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
    return frame


def KDLFromString(str, delimiter=' ', fmt='RPY'):

    if fmt != 'RPY':
        print("Format {} not supported yet!".format(fmt))
        return PyKDL.Frame()

    chunks = map(float, str.split(delimiter))
    frame = PyKDL.Frame()
    frame.p = PyKDL.Vector(
        chunks[0], chunks[1], chunks[2]
    )
    frame.M = PyKDL.Rotation.RPY(
        chunks[3],
        chunks[4],
        chunks[5]
    )
    return frame


def KDLToCv(frame):
    """ Converts the OpenCV couple to PyKDL Frame """
    rot = numpy.array([
        [frame.M[0, 0], frame.M[0, 1], frame.M[0, 2]],
        [frame.M[1, 0], frame.M[1, 1], frame.M[1, 2]],
        [frame.M[2, 0], frame.M[2, 1], frame.M[2, 2]]
    ]
    )
    Rvec, _ = cv2.Rodrigues(rot)
    Tvec = numpy.array([frame.p.x(), frame.p.y(), frame.p.z()])

    return Rvec, Tvec


def KDLFrom2DRF(vx, vy, center):
    rot = PyKDL.Rotation(
        vx[0], vy[0], 0.0,
        vx[1], vy[1], 0.0,
        0, 0, -1
    )
    frame = PyKDL.Frame()
    frame.M = rot
    frame.p.x(center[0])
    frame.p.y(center[1])
    return frame


def KDLFromRPY(roll, pitch, yaw):
    frame = PyKDL.Frame()
    frame.M = PyKDL.Rotation.RPY(roll, pitch, yaw)
    return frame


#########################################################################
#########################################################################
#########################################################################
#########################################################################


def KDLVectorToNumpyArray(vector):
    """ Transform a KDL vector to a Numpy Array """
    return numpy.array(
        [vector.x(), vector.y(), vector.z()]
    ).reshape(3)


def planeCoefficientsFromFrame(frame):
    """ Builds 3D Plane coefficients centered in a Reference Frame """

    normal = frame.M.UnitZ()
    a = normal.x()
    b = normal.y()
    c = normal.z()
    d = -(a * frame.p.x() + b * frame.p.y() + c * frame.p.z())
    return numpy.array([a, b, c, d]).reshape(4)


def cloneFrame(frame):
    f2 = PyKDL.Frame()
    f2.p = frame.p
    f2.M = frame.M
    return f2


def frameDistance(frame1, frame2):
    """ Distance between two frames. [translational_distance, quaternion_distance] """
    dist_t = frame1.p - frame2.p
    q1 = numpy.array(frame1.M.GetQuaternion())
    q2 = numpy.array(frame2.M.GetQuaternion())
    q1 = q1 / numpy.linalg.norm(q1)
    q2 = q2 / numpy.linalg.norm(q2)
    dist_a = 0.0
    for i in range(0, 4):
        dist_a += q1[i] * q2[i]
    dist_a = 1 - dist_a * dist_a
    return [dist_t, dist_a]


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis."""

    data = numpy.array(data, dtype=numpy.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(numpy.dot(data, data))
        data *= data
        out = numpy.atleast_1d(numpy.sum(data, axis=axis))
        numpy.sqrt(out, out)
        return out
    else:
        data *= data
        numpy.sum(data, axis=axis, out=out)
        numpy.sqrt(out, out)


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.

    >>> q = quaternion_about_axis(0.123, [1, 0, 0])
    >>> numpy.allclose(q, [0.99810947, 0.06146124, 0, 0])
    True

    """
    qx = axis[0] * math.sin(angle / 2)
    qy = axis[1] * math.sin(angle / 2)
    qz = axis[2] * math.sin(angle / 2)
    qw = math.cos(angle / 2)
    return numpy.array([qx, qy, qz, qw])


def quaternion_inverse(quaternion):
    """Return inverse of quaternion."""
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    numpy.negative(q[1:], q[1:])
    return q / numpy.dot(q, q)


def quaternion_angle(quaternion):
    return math.acos(quaternion[3]) * 2.0


def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions."""
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return numpy.array([
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=numpy.float64)


def perpendicular_vector(v):
    """ Finds an arbitrary perpendicular vector to * v * ."""
    # x = y = z = 0 is not an acceptable solution
    if v.x() == v.y() == v.z() == 0:
        raise ValueError('zero-vector')

    if v.x() == 0:
        return PyKDL.Vector(1, 0, 0)
    if v.y() == 0:
        return PyKDL.Vector(0, 1, 0)
    if v.z() == 0:
        return PyKDL.Vector(0, 0, 1)

    # arbitrarily set a = b = 1
    # then the equation simplifies to
    #     c = -(x + y)/z
    return PyKDL.Vector(1, 1, -1.0 * (v.x() + v.y()) / v.z())

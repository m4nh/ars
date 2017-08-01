#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pkgutil
import rospy
import tf
import rospkg
import os
import roars.geometry.transformations as transformations
from roars.rosutils.logger import Logger


class RosParamType(object):
    PRIVATE_PARAM = 1
    GLOBAL_PARAM = 10

##########################################################################
##########################################################################
##########################################################################
##########################################################################


class RosNode(object):

    def __init__(self, node_name="new_node", hz=30):
        self.node_name = node_name
        self.hz = hz
        self._node = rospy.init_node(self.node_name)

        if self.hz > 0:
            self.rate = rospy.Rate(self.hz)
        else:
            self.rate = None

        #⬢⬢⬢⬢⬢➤ Parameters map
        self._parameters = {}

        #⬢⬢⬢⬢⬢➤ Topics
        self.publishers = {}
        self.subsribers = {}

        #⬢⬢⬢⬢⬢➤ helpers
        self.tf_listener = None
        self.tf_broadcaster = None
        self.rospack = rospkg.RosPack()

        #⬢⬢⬢⬢⬢➤ Timings
        self.starting_time = -1

    def getName(self, replace_slash=True):
        name = rospy.get_name()
        if replace_slash:
            name = name.replace("/", "")
        return name

    def setHz(self, hz):
        self.hz = hz
        self.rate = rospy.Rate(self.hz)

    def getRosNode(self):
        return self._node

    def createSubscriber(self, topic_name, msg_type, callback, queue_size=1, user_name=None):
        if user_name == None:
            user_name = topic_name
        self.subsribers[topic_name] = rospy.Subscriber(
            topic_name,
            msg_type,
            callback,
            queue_size=queue_size
        )
        return self.subsribers[topic_name]

    def createPublisher(self, topic_name, msg_type, queue_size=1, user_name=None):
        if user_name == None:
            user_name = topic_name
        self.publishers[topic_name] = rospy.Publisher(
            topic_name,
            msg_type,
            queue_size=queue_size
        )
        return self.publishers[topic_name]

    def getPublisher(self, name):
        if name in self.publishers:
            return self.publishers[name]
        return None

    def getSubscriber(self, name):
        if name in self.subsribers:
            return self.subsribers[name]
        return None

    def _sleep(self):
        self.rate.sleep()

    def tick(self):
        if self.starting_time < 0:
            if self.getCurrentTime().to_sec() > 0:
                self.starting_time = self.getCurrentTime().to_sec()
        self._sleep()

    def isActive(self):
        # if self.starting_time < 0:
        #    self.tick()
        return not rospy.is_shutdown()

    def getTFListener(self):
        if self.tf_listener == None:
            self.tf_listener = tf.TransformListener()
        return self.tf_listener

    def getTFBroadcaster(self):
        if self.tf_broadcaster == None:
            self.tf_broadcaster = tf.TransformBroadcaster()
        return self.tf_broadcaster

    def getCurrentTime(self):
        return rospy.Time.now()

    def getCurrentTimeInSecs(self):
        return self.getCurrentTime().to_sec()

    def getElapsedTimeInSecs(self):
        if self.starting_time < 0:
            return 0
        return self.getCurrentTimeInSecs() - self.starting_time

    def broadcastTransform(self, frame, frame_id, parent_frame_id, time):
        transformations.broadcastTransform(
            self.getTFBroadcaster(),
            frame,
            frame_id,
            parent_frame_id,
            time
        )

    def retrieveTransform(self, frame_id, parent_frame_id, time):
        if time == -1:
            time = rospy.Time(0)
        return transformations.retrieveTransform(self.getTFListener(), parent_frame_id, frame_id, time, none_error=True, print_error=True)

    def getFileInPackage(self, pkg_name, file_path):
        pack_path = self.rospack.get_path(pkg_name)
        return os.path.join(pack_path, file_path)

    def getRosParameter(self, parameter_name, default_value, type=RosParamType.PRIVATE_PARAM):
        final_name = parameter_name
        if type == RosParamType.PRIVATE_PARAM:
            final_name = '~' + final_name
        return rospy.get_param(final_name, default_value)

    def setupParameter(self, parameter_name, default_value, type=RosParamType.PRIVATE_PARAM, array_type=None):
        par = self.getRosParameter(parameter_name, default_value, type=type)
        if array_type != None:
            par = map(array_type, par.split(";"))
        self._parameters[parameter_name] = par
        return par

    def getParameter(self, parameter_name):
        if parameter_name in self._parameters:
            return self._parameters[parameter_name]
        return None

    def getLogger(self):
        return Logger

    def close(self, reason=0):
        rospy.signal_shutdown(reason)

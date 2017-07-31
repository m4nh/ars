#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.rosutils.rosnode import RosNode

#⬢⬢⬢⬢⬢➤ NODE
node = RosNode("rosnode_example")

#⬢⬢⬢⬢⬢➤ Sets HZ from parameters
node.setHz(node.setupParameter("hz", 1))

while node.isActive():
    node.getLogger().log("Time: {}".format(node.getCurrentTimeInSecs()))
    node.tick()

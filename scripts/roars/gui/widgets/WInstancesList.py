#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.gui.pyqtutils import PyQtWidget, PyQtImageConverter
from roars.datasets.datasetutils import TrainingInstance, TrainingClass
from WBaseWidget import WBaseWidget
from WAxesEditor import WAxesEditor
from WAxesButtons import WAxesButtons
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4 import QtCore
import PyQt4.QtGui as QtGui
import math


class WInstancesList(WBaseWidget):

    def __init__(self, changeCallback=None, newCallback=None):
        super(WInstancesList, self).__init__(
            'ui_instances_list'
        )

        self.instances = []

        #⬢⬢⬢⬢⬢➤ Change Callback
        self.changeCallback = changeCallback
        self.newCallback = newCallback
        if self.newCallback:
            self.ui_list.doubleClicked.connect(self.newCallback)

    def getSelectedInstance(self):
        return self.instances[self.selected_instance]

    def getInstances(self):
        return self.instances

    def setInstances(self, instances):
        self.instances = instances
        self.refreshInstacesList(self.instances)

    def refreshInstacesList(self, instances):
        list_model = QStandardItemModel(self.ui_list)
        list_model.clear()
        for i in range(0, len(instances)):
            inst = instances[i]
            item = QStandardItem()
            item.setText("Instance_{}".format(i))
            item.setCheckable(False)
            list_model.appendRow(item)

        self.ui_list.setModel(list_model)
        self.ui_list.selectionModel().currentChanged.connect(
            self.listChange)

    def listChange(self, current, previous):
        self.selected_instance = current.row()
        if self.selected_instance >= 0:
            inst = self.instances[self.selected_instance]
            if self.changeCallback != None:
                self.changeCallback(inst)

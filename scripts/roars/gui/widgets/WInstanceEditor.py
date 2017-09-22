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


class WInstanceEditor(WBaseWidget):

    def __init__(self, changeCallback=None):
        super(WInstanceEditor, self).__init__(
            'ui_instance_editor'
        )

        self.ui_axis_translation = WAxesEditor(names=['cx', 'cy', 'cz'], labels=[
                                               'X', 'Y', 'Z'], unit='m', step=0.001, changeCallback=self.valueChanged)
        self.ui_axis_rotation = WAxesEditor(names=['roll', 'pitch', 'yaw'], labels=[
                                            'R', 'P', 'Y'], unit=u'\N{DEGREE SIGN}', step=0.2, binding_conversion=180.0 / math.pi, changeCallback=self.valueChanged)
        self.ui_axes_container.addWidget(self.ui_axis_translation)
        self.ui_axes_container.addWidget(self.ui_axis_rotation)

        self.ui_axes_relative_buttons = WAxesButtons(
            name='relative', label='Relative Movement', changeCallback=self.axesButtonCallback)
        self.ui_axes_size_butttons = WAxesButtons(
            name='size', label='Size', changeCallback=self.axesButtonCallback)

        self.ui_relative_axes_container.addWidget(
            self.ui_axes_relative_buttons)
        self.ui_relative_axes_container.addWidget(self.ui_axes_size_butttons)

        #⬢⬢⬢⬢⬢➤ model
        self.instance = None

        #⬢⬢⬢⬢⬢➤ Change Callback
        self.changeCallback = changeCallback

        #⬢⬢⬢⬢⬢➤ Classes
        self.class_map = {}
        self.ui_class_list.currentIndexChanged.connect(self.classListChange)

    def axesButtonCallback(self, data):
        if self.instance != None:
            tp = data[0]
            axis = data[1]
            delta = data[2]
            if tp == 'relative':
                self.instance.relativeTranslations(axis, delta)
            if tp == 'size':
                self.instance.grows(axis, delta)

        self.notifyChanges()
        self.updateUI()

    def setSelectedInstance(self, instance):
        self.instance = instance
        self.updateUI()

    def updateUI(self):
        if self.instance != None:
            properties = self.instance.getFrameProperties()
            for name, value in properties.iteritems():
                self.ui_axis_translation.setValue(name, value)
                self.ui_axis_rotation.setValue(name, value)
            self.ui_class_list.setCurrentIndex(self.instance.label + 1)

    def valueChanged(self, data):
        if self.instance != None:
            for name, value in data.iteritems():
                self.instance.setFrameProperty(name, value)
            self.notifyChanges()

    def notifyChanges(self):
        if self.changeCallback != None:
            self.changeCallback()

    def setClassMap(self, class_map):
        self.class_map = class_map
        self.ui_class_list.clear()
        model = self.ui_class_list.model()
        labels = sorted(class_map.keys())

        for k in labels:
            v = class_map[k]
            item = QtGui.QStandardItem(v.name)
            color = TrainingClass.getColorByLabel(k)
            item.setForeground(QtGui.QColor(color[2], color[1], color[0]))
            font = item.font()
            font.setPointSize(10)
            item.setFont(font)
            model.appendRow(item)

    def classListChange(self, index):
        combo = self.sender()
        color = TrainingClass.getColorByLabel(index - 1, output_type="HEX")
        combo.setStyleSheet(
            "QComboBox::drop-down {background: " + color + ";}")
        if self.instance != None:
            self.instance.label = index - 1
        self.notifyChanges()

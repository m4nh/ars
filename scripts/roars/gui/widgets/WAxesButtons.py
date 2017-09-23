#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.gui.pyqtutils import PyQtWidget, PyQtImageConverter
from WBaseWidget import WBaseWidget
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4 import QtCore


class WAxesButtons(WBaseWidget):

    def __init__(self, name='axes', label='Axes Buttons', changeCallback=None, step=0.001):
        super(WAxesButtons, self).__init__(
            'ui_axes_buttons'
        )

        self.name = name
        self.label = label
        self.step = step
        self.ui_label.setText(label)

        #⬢⬢⬢⬢⬢➤ Callback
        self.changeCallback = changeCallback

        self.buttons = {
            'x+': self.ui_button_x_plus,
            'x-': self.ui_button_x_minus,
            'y+': self.ui_button_y_plus,
            'y-': self.ui_button_y_minus,
            'z+': self.ui_button_z_plus,
            'z-': self.ui_button_z_minus
        }
        self.buttons_name_map = {}

        # TODO:example style
        self.ui_button_x_minus.setStyleSheet(
            "QPushButton:hover{background-color: red}")

        for label, button in self.buttons.iteritems():
            button.clicked.connect(self.buttonPressed)
            self.buttons_name_map[str(button.objectName())] = label

    def buttonPressed(self):
        if self.changeCallback != None:
            label = self.buttons_name_map[str(self.sender().objectName())]
            delta = float(label[1] + str(self.step))
            val = (self.name, label[0], delta)
            self.changeCallback(val)

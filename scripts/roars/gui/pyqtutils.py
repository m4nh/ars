#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys
import signal
import numpy as np

#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################


class PyQtNamesMap(object):
    class_map = {
        "QPushButton": "button"
    }

    @staticmethod
    def getClassName(qt_name):
        if qt_name in PyQtNamesMap.class_map:
            return PyQtNamesMap.class_map[qt_name]
        return qt_name

#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################


class PyQtImageConverter(object):

    class NotImplementedException:
        pass

    gray_color_table = [qRgb(i, i, i) for i in range(256)]

    @staticmethod
    def cvToQPixmap(im, copy=False):
        qimg = PyQtImageConverter.cvToQImage(im, copy=copy)
        return QtGui.QPixmap.fromImage(qimg)

    @staticmethod
    def cvToQImage(im, copy=False):
        if im is None:
            return QtGui.QImage()

        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QtGui.QImage(
                    im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_Indexed8)

                qim.setColorTable(PyQtImageConverter.gray_color_table)

                return qim.copy() if copy else qim

            elif len(im.shape) == 3:
                if im.shape[2] == 1:
                    qim = QtGui.QImage(
                        im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_Indexed8)
                    qim.setColorTable(PyQtImageConverter.gray_color_table)
                    return qim.copy() if copy else qim
                elif im.shape[2] == 3:
                    qim = QtGui.QImage(
                        im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QtGui.QImage(
                        im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32)
                    return qim.copy() if copy else qim

        raise PyQtImageConverter.NotImplementedException

#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################


class PyQtWindow(QtGui.QMainWindow):
    def __init__(self, uifile, namespace="default"):
        self.qt_application = QApplication(sys.argv)
        super(PyQtWindow, self).__init__()
        uic.loadUi(uifile, self)

        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def run(self):
        self.show()
        sys.exit(self.qt_application.exec_())

    def showDialog(self, text='', title="Info", info="", details="", buttons=QMessageBox.Ok | QMessageBox.Cancel, callback=None):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        msg.setInformativeText(info)
        msg.setWindowTitle(title)
        msg.setDetailedText(details)
        msg.setStandardButtons(buttons)
        if callback:
            msg.buttonClicked.connect(callback)

        retval = msg.exec_()
        return retval

    def showPromptBool(self, title='', message='', yes_msg=QtGui.QMessageBox.Yes, no_msg=QtGui.QMessageBox.No):
        reply = QtGui.QMessageBox.question(
            self,
            title,
            message,
            yes_msg,
            no_msg
        )
        return reply == QtGui.QMessageBox.Yes

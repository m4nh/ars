#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from roars.gui.pyqtutils import PyQtWidget, PyQtImageConverter
from WBaseWidget import WBaseWidget
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from PyQt4 import QtCore



class WAxesEditor(WBaseWidget):

    def __init__(self,names=['x','y','z'], labels = ['Axis_x','Axis_y','Axis_z'],unit='m',step=0.001,binding_conversion=1,changeCallback=None):
        super(WAxesEditor, self).__init__(
            'ui_axes_editor'
        )
        
        self.names = names
        self.spins = {
            self.names[0]: self.ui_spin_x,
            self.names[1]: self.ui_spin_y,
            self.names[2]: self.ui_spin_z
        }
        self.spins_name_map = {}

        for label,spin in self.spins.iteritems():
            spin.valueChanged.connect(self.valueChanged)
            self.spins_name_map[str(spin.objectName())] = label

        #⬢⬢⬢⬢⬢➤ Callback
        self.labels = labels
        self.changeCallback = changeCallback

        #⬢⬢⬢⬢⬢➤ Set Labels
        self.ui_label_x.setText( self.labels[0])
        self.ui_label_y.setText( self.labels[1])
        self.ui_label_z.setText( self.labels[2])

        #⬢⬢⬢⬢⬢➤ Set Parameters
        self.binding_conversion = binding_conversion
        self.step = step
        self.setStep(step)
        self.unit = unit
        self.setUnit(unit)

    def setUnit(self,unit):
        for label,spin in self.spins.iteritems():
            spin.setSuffix(' '+unit)

    def setStep(self,step):
        for label,spin in self.spins.iteritems():
            spin.setSingleStep(step)
            spin.setDecimals(str(step)[::-1].find('.'))
            spin.setRange(-1000000.0,1000000.0)

    def valueChanged(self,v):
        if self.changeCallback != None:
            self.changeCallback(self.getValues())

    def setValue(self,label,v):
        if label in self.spins:
            self.spins[label].setValue(v*self.binding_conversion)

    def setValues(self,values):
        if len(values)>=len(self.names):
            for i in range(0,len(self.names)):
                self.setValue(self.names[i],values[i])
    
    def getValue(self,label):
        if label in self.spins:
            return self.spins[label].value()/self.binding_conversion
        return None

    def getValues(self):
        values = {}
        for i in range(0,len(self.names)):
            values[self.names[i]] =self.getValue(self.names[i])
        return values    


       

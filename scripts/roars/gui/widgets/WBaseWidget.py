from roars.gui.pyqtutils import PyQtWidget
from roars.datasets.datasetutils import TrainingScene
import os


class WBaseWidget(PyQtWidget):
    DEFAULT_UI_WIDGETS_FOLDER = ''

    def __init__(self, uiname):
        super(WBaseWidget, self).__init__(
            uifile=os.path.join(
                WBaseWidget.DEFAULT_UI_WIDGETS_FOLDER,
                uiname + ".ui"
            )
        )

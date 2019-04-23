from PyQt5.QtCore import (QThread, pyqtSignal)

class BgFunctionWorker(QThread):
    fn_finished = pyqtSignal(object)
    def __init__(self, parent = None, fn = None):
        QThread.__init__(self, parent)
        self.fn = fn

    def run(self):
        res = self.fn()
        self.fn_finished.emit(res)
        

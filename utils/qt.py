import cv2
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QPixmap

def opencv2qtpixmap(opencv_frame, width_px = 640, height_px = 480):
    rgbImage = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB)
    convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],
                                        QtGui.QImage.Format_RGB888)
    convertToQtFormat = QtGui.QPixmap.fromImage(convertToQtFormat)
    pixmap = QPixmap(convertToQtFormat)
    return pixmap.scaled(width_px, height_px, QtCore.Qt.KeepAspectRatio)
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/Main.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(231, 219)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.lblImage = QtWidgets.QLabel(self.centralwidget)
        self.lblImage.setText("")
        self.lblImage.setObjectName("lblImage")
        self.verticalLayout.addWidget(self.lblImage)
        self.gridLayout.addLayout(self.verticalLayout, 4, 0, 1, 1)
        self.btnBrowse = QtWidgets.QPushButton(self.centralwidget)
        self.btnBrowse.setObjectName("btnBrowse")
        self.gridLayout.addWidget(self.btnBrowse, 1, 0, 1, 1)
        self.btnAnimate = QtWidgets.QPushButton(self.centralwidget)
        self.btnAnimate.setObjectName("btnAnimate")
        self.gridLayout.addWidget(self.btnAnimate, 2, 0, 1, 1)
        self.pbarAnimation = QtWidgets.QProgressBar(self.centralwidget)
        self.pbarAnimation.setProperty("value", 0)
        self.pbarAnimation.setTextVisible(True)
        self.pbarAnimation.setObjectName("pbarAnimation")
        self.gridLayout.addWidget(self.pbarAnimation, 3, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "KeypointRecognizer"))
        self.lblImage.setText(_translate("MainWindow", " "))
        self.btnBrowse.setText(_translate("MainWindow", "Browse..."))
        self.btnAnimate.setText(_translate("MainWindow", "Animate"))


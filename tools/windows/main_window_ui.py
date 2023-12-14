# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.5.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(980, 700)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.output_mood = QLabel(self.centralwidget)
        self.output_mood.setObjectName(u"output_mood")
        self.output_mood.setGeometry(QRect(90, 20, 391, 291))
        self.output_mood.setScaledContents(True)
        self.output_mood.setAlignment(Qt.AlignCenter)
        self.output_sta = QLabel(self.centralwidget)
        self.output_sta.setObjectName(u"output_sta")
        self.output_sta.setGeometry(QRect(500, 20, 391, 291))
        self.output_sta.setScaledContents(True)
        self.output_sta.setAlignment(Qt.AlignCenter)
        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(480, 0, 20, 321))
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.input_ori = QLabel(self.centralwidget)
        self.input_ori.setObjectName(u"input_ori")
        self.input_ori.setGeometry(QRect(290, 330, 391, 291))
        self.input_ori.setScaledContents(True)
        self.input_ori.setAlignment(Qt.AlignCenter)
        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(60, 310, 881, 20))
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.det_head = QPushButton(self.centralwidget)
        self.det_head.setObjectName(u"det_head")
        self.det_head.setGeometry(QRect(210, 630, 271, 41))
        self.det_ori = QPushButton(self.centralwidget)
        self.det_ori.setObjectName(u"det_ori")
        self.det_ori.setGeometry(QRect(500, 630, 271, 41))
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.output_mood.setText(QCoreApplication.translate("MainWindow", u"\u60c5\u7eea\u68c0\u6d4b", None))
        self.output_sta.setText(QCoreApplication.translate("MainWindow", u"\u884c\u4e3a\u68c0\u6d4b", None))
        self.input_ori.setText(QCoreApplication.translate("MainWindow", u"\u663e\u793a\u539f\u59cb\u89c6\u9891", None))
        self.det_head.setText(QCoreApplication.translate("MainWindow", u"\u6444\u50cf\u5934\u89c6\u9891", None))
        self.det_ori.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4f20\u89c6\u9891", None))
    # retranslateUi


import sys
import torch
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog
from PySide6.QtGui import QPixmap, QImage
import cv2
from PySide6.QtCore import QTimer
from main_window_ui import Ui_MainWindow
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt 
def convert2QImage(img):
    height, width, channel = img.shape
    return QImage(img, width, height, width * channel, QImage.Format_RGB888)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.model_status = YOLO(r"runs\detect\train40\weights\best.pt", task='detect')
        self.model_mood = YOLO(r"runs\detect\train39\weights\best.pt", task='detect')
        self.timer = QTimer()
        self.timer.setInterval(1)
        self.video = None
        self.bind_slots()

    def video_pred(self):
        ret, frame = self.video.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            self.timer.stop()
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.input_ori.setPixmap(QPixmap.fromImage(convert2QImage(frame)))
            results_mood = self.model_mood(frame, conf=0.2, iou=0.8)
            results_status = self.model_status(frame, conf=0.2, iou=0.8)
            image_mood = results_mood[0].plot()
            
            image_status = results_status[0].plot()
            self.output_mood.setPixmap(QPixmap.fromImage(convert2QImage(image_mood)))
            self.output_sta.setPixmap(QPixmap.fromImage(convert2QImage(image_status)))
        
    def open_video(self):
        file_path = QFileDialog.getOpenFileName(self, dir="./datasets", filter="*.mp4")
        if file_path[0]:
            file_path = file_path[0]
            self.video = cv2.VideoCapture(file_path)
            self.timer.start()
            
    def open_head(self):
        self.video = cv2.VideoCapture(0)
        self.timer.start()

    def bind_slots(self):
        self.det_head.clicked.connect(self.open_head)
        self.det_ori.clicked.connect(self.open_video)
        self.timer.timeout.connect(self.video_pred)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
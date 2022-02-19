#!/usr/bin/python
# -*- coding:utf-8 -*-
from PyQt5.Qt import *
from PyQt5.QtCore import Qt
import cv2
import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from PyQt5 import QtCore, QtWidgets
import sys
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box

class GUI_view(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None

        # 去边框并设置背景图片及大小
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.resize(1131, 650)
        window_pale = QtGui.QPalette()
        window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap("bg_imgs/background.jpg")))
        self.setPalette(window_pale)

        # 重写“最小化”、“缩小界面”、“关闭”按钮
        self.button_red = QPushButton('×', self)
        self.button_red.move(1050, 11)
        self.button_red.setFixedSize(18, 18)
        self.button_red.setStyleSheet(
            "QPushButton{background:#CE0000;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:red;}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        self.button_red.clicked.connect(QCoreApplication.quit)
        # self.button_orange = QPushButton(self)
        # self.button_orange.move(1010, 11)
        # self.button_orange.setFixedSize(18, 18)
        # self.button_orange.setStyleSheet(
        #     "QPushButton{background:orange;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
        #     "QPushButton:hover{background:#FFD306}"
        #     "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        # self.button_green = QPushButton(self)
        # self.button_green.move(970, 11)
        # self.button_green.setFixedSize(18, 18)
        # self.button_green.setStyleSheet(
        #     "QPushButton{background:green;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
        #     "QPushButton:hover{background:#08BF14}"
        #     "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=r'.\weights\best.pt', help='model.pt path(s)')
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        print(self.opt)

        weights, view_img, save_txt, imgsz = self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        self.device = 'cuda'
        self.half = True
        cudnn.benchmark = True

        self.model = attempt_load(weights, map_location=self.device)
        stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz, s=stride)
        if self.half:
            self.model.half()

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def setupUi(self, MainWindow):
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.pushButton_img = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_img.move(180, 100)
        self.pushButton_img.setFixedSize(188, 88)
        self.pushButton_img.setStyleSheet(
            "QPushButton{background:#0080FF;color:white;font-size:20px;border-radius:24px;font-weight:bold;font-family: YaHei}"
            "QPushButton:hover{background:#2894FF}"
            "QPushButton:pressed{background:#004B97}")
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_camera.move(480, 100)
        self.pushButton_camera.setFixedSize(188, 88)
        self.pushButton_camera.setStyleSheet(
            "QPushButton{background:#0080FF;color:white;font-size:20px;border-radius:24px;font-weight:bold;font-family: YaHei}"
            "QPushButton:hover{background:#2894FF}"
            "QPushButton:pressed{background:#004B97}")
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_video.move(780, 100)
        self.pushButton_video.setFixedSize(188, 88)
        self.pushButton_video.setStyleSheet(
            "QPushButton{background:#0080FF;color:white;font-size:20px;border-radius:24px;font-weight:bold;font-family: YaHei}"
            "QPushButton:hover{background:#2894FF}"
            "QPushButton:pressed{background:#004B97}")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.move(199, 211)
        self.label.setFixedSize(752, 400)
        self.label.setStyleSheet(
            "QPushButton{background:#0080FF;color:white;font-size:20px;border-radius:24px;font-weight:bold;font-family: YaHei}"
            "QPushButton:hover{background:#2894FF}"
            "QPushButton:pressed{background:#004B97}")
        MainWindow.setCentralWidget(self.centralwidget)
        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(811, 433)
        self.classlabel.move(172, 199)
        self.classlabel.setPixmap(QPixmap("bg_imgs/biankuang.png"))
        self.classlabel.setScaledContents(True)
        self.classlabel = QLabel(self)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.pushButton_img.setText(_translate("MainWindow", "图片检测"))
        self.pushButton_camera.setText(_translate("MainWindow", "摄像头检测"))
        self.pushButton_video.setText(_translate("MainWindow", "视频检测"))
        self.label.setText(_translate("MainWindow", "TextLabel"))

    def init_slots(self):
        self.pushButton_img.clicked.connect(self.button_image_open)
        self.pushButton_video.clicked.connect(self.button_video_open)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        self.timer_video.timeout.connect(self.show_video_frame)

    def init_logo(self):
        pix = QtGui.QPixmap('wechat.jpg')
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    def button_image_open(self):
        print('button_image_open')
        name_list = []

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        img = cv2.imread(img_name)
        print(img_name)
        showimg = img
        num = 0
        with torch.no_grad():
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=self.opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            print(pred)
            # Process detections
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        num = num + 1
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        name_list.append(self.names[int(cls)])
                        plot_one_box(xyxy, showimg, label=None, color=(255, 255, 0), line_thickness=1)

        num = 'num:' + str(num)
        cv2.putText(showimg, str(num), (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 1)
        cv2.imwrite('prediction.jpg', showimg)
        self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                  QtGui.QImage.Format_RGB32)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")
        flag = self.cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20,
                                       (int(self.cap.get(3)), int(self.cap.get(4))))
            self.timer_video.start(30)
            self.pushButton_video.setDisabled(True)
            self.pushButton_img.setDisabled(True)
            self.pushButton_camera.setDisabled(True)

    def button_camera_open(self):
        if not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.cap.open(0)
            if flag == False:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20,
                                           (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)
                self.pushButton_video.setDisabled(True)
                self.pushButton_img.setDisabled(True)
                self.pushButton_camera.setText(u"关闭摄像头")
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.init_logo()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setText(u"摄像头检测")

    def show_video_frame(self):
        name_list = []
        flag, img = self.cap.read()
        num = 0
        if img is not None:
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            num = num + 1
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            plot_one_box(xyxy, showimg, label=None, color=(255, 255, 0), line_thickness=1)

                        num = 'num:' + str(num)
                        cv2.putText(showimg, str(num), (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 1)

            self.out.write(showimg)
            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setDisabled(False)
            self.init_logo()

    def quit_button(self):
        quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GUI_view()
    window.show()
    sys.exit(app.exec())

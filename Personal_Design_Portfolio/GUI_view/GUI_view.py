#!/usr/bin/python
# -*- coding:utf-8 -*-
from PyQt5.Qt import *
from PyQt5.QtCore import Qt
from PyQt5.QtWebEngineWidgets import *
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
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box
from mobilenet_test import detect
from get_class_value import query
from get_elimination_strategy import elimination_strategy

#-------------------------------------------------#
#  登录界面
#-------------------------------------------------#
class LoginWindow(QWidget):
    def __init__(self):
        super(LoginWindow, self).__init__()

        self.setWindowFlags(Qt.FramelessWindowHint)  # 去边框
        self.resize(1169, 731)   # resize
        window_pale = QtGui.QPalette()
        window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap("images/background.jpg")))  # 设定背景图
        self.setPalette(window_pale)

        # -------------------------------------------------#
        #  载入title_pic
        # -------------------------------------------------#
        self.titlelabel = QLabel(self)
        self.titlelabel.setFixedSize(130, 30)
        self.titlelabel.move(525, 167)
        self.titlelabel.setPixmap(QPixmap("images/title.png"))
        self.titlelabel.setScaledContents(True)

        # -------------------------------------------------#
        #  载入logo_pic
        # -------------------------------------------------#
        self.logolabel = QLabel(self)
        self.logolabel.setFixedSize(111, 99)
        self.logolabel.move(199, 50)
        self.logolabel.setPixmap(QPixmap("images/logo_log.png"))
        self.logolabel.setScaledContents(True)

        # -------------------------------------------------#
        #  载入name_pic
        # -------------------------------------------------#
        self.namelabel = QLabel(self)
        self.namelabel.setFixedSize(700, 120)
        self.namelabel.move(300, 40)
        self.namelabel.setPixmap(QPixmap("images/name.png"))
        self.namelabel.setScaledContents(True)

        # -------------------------------------------------
        #  设置用户名
        # -------------------------------------------------
        self.le0 = QLineEdit(self)
        self.le0.move(440, 235)
        self.le0.resize(300, 38)
        self.le0.setPlaceholderText(" User Name")
        self.le0.setStyleSheet("background:transparent;border:0px;font-size:21px;color:white")
        self.le0.setClearButtonEnabled(True)

        # -------------------------------------------------
        #  设置密码
        # -------------------------------------------------
        self.le1 = QLineEdit(self)
        self.le1.move(440, 303)
        self.le1.resize(300, 38)
        self.le1.setPlaceholderText(" Password")
        self.le1.setStyleSheet("background:transparent;border:0px;font-size:21px;color:white")
        self.le1.setClearButtonEnabled(True)
        self.le1.setEchoMode(QLineEdit.Password)

        # -------------------------------------------------
        #  设置“登录系统”按钮点击事件
        # -------------------------------------------------
        loginbtn = QPushButton(self)
        loginbtn.move(405, 424)
        loginbtn.resize(363, 45)
        loginbtn.clicked.connect(self.onLoginClick)
        loginbtn.setStyleSheet("background:transparent;border:0px")
        loginbtn.setCursor(QCursor(Qt.PointingHandCursor))

        # -------------------------------------------------
        #  设置“账号注册”按钮点击事件
        # -------------------------------------------------
        register_btn = QPushButton(self)
        register_btn.move(400, 370)
        register_btn.setFixedSize(77, 33)
        register_btn.setStyleSheet("background:transparent;border:0px")
        register_btn.setCursor(QCursor(Qt.PointingHandCursor))
        register_btn.clicked.connect(self.register_btn)

        # -------------------------------------------------
        #  设置“忘记密码”按钮点击事件
        # -------------------------------------------------
        forget_password_btn = QPushButton(self)
        forget_password_btn.move(700, 370)
        forget_password_btn.setFixedSize(77, 33)
        forget_password_btn.setStyleSheet("background:transparent;border:0px")
        forget_password_btn.setCursor(QCursor(Qt.PointingHandCursor))
        forget_password_btn.clicked.connect(self.forget_password)

        # -------------------------------------------------
        #  设置“版权”信息
        # -------------------------------------------------
        self.copyrightlabel = QLabel(self)
        self.copyrightlabel.setText("             Copyright @ 2021.09 · 赵泽荣 \n From Software College, Shanxi Agricultural University")
        self.copyrightlabel.move(340, 633)
        self.copyrightlabel.setStyleSheet(
            "QLabel{color:#E0E0E0;font-size:18px;}"
        )

    # -------------------------------------------------
    #  设置点击“账号注册”按钮弹出事件
    # -------------------------------------------------
    def register_btn(self):
        QMessageBox.information(self, "抱歉", "该功能正在抢修中...")

    # -------------------------------------------------
    #  设置点击“忘记密码”按钮弹出事件
    # -------------------------------------------------
    def forget_password(self):
        QMessageBox.information(self, "抱歉", "该功能正在抢修中...")

    # -------------------------------------------------
    #  设置点击“登录系统”按钮弹出事件
    # -------------------------------------------------
    def onLoginClick(self):
        print('信息核验成功，正在进入系统...')
        qApp = QApplication.instance()
        qApp.quit()  # 关闭窗口

    # -------------------------------------------------
    #  重写鼠标事件
    # -------------------------------------------------
    # 重写鼠标移动事件
    def mouseMoveEvent(self, e: QMouseEvent):
        if self._tracking:
            self._endPos = e.pos() - self._startPos
            self.move(self.pos() + self._endPos)
            self.setCursor(QCursor(Qt.SizeAllCursor))
    # 重写鼠标“按”事件
    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._startPos = QPoint(e.x(), e.y())
            self._tracking = True
    # 重写鼠标“抬”事件
    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._tracking = False
            self._startPos = None
            self._endPos = None
            self.setCursor(QCursor(Qt.ArrowCursor))

# -------------------------------------------------
#  设置登录界面后首先展现的系统界面信息
# -------------------------------------------------
class function1_view(QWidget):
    def __init__(self):
        super().__init__()
        # -------------------------------------------------
        #  展示logo及标题pic
        # -------------------------------------------------
        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(99, 99)
        self.classlabel.move(143, 80)
        self.classlabel.setPixmap(QPixmap("images/logo.png"))
        self.classlabel.setScaledContents(True)
        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(588, 101)
        self.classlabel.move(244, 77)
        self.classlabel.setPixmap(QPixmap("images/name.png"))
        self.classlabel.setScaledContents(True)

        # -------------------------------------------------
        #  展示版本号信息
        # -------------------------------------------------
        self.classlabel = QLabel(self)
        self.classlabel.setText("版本号：V-2.0.1")
        self.classlabel.move(666, 166)
        self.classlabel.setStyleSheet(
            "QLabel{color:#424200;font-size:17px;font-weight:bold;font-family:KaiTi;}"
        )

        # -------------------------------------------------
        #  因关闭边框，模拟mac系统重写“最小化”、“缩小界面”、“关闭”按钮
        # -------------------------------------------------
        # 红色按钮：重写“关闭”事件
        self.button_red = QPushButton(self)
        self.button_red.move(900, 11)
        self.button_red.setFixedSize(18, 18)
        self.button_red.setStyleSheet("QPushButton{background:#CE0000;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
                                      "QPushButton:hover{background:red;}"
                                      "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        self.button_red.clicked.connect(self.quit_button)
        # 黄色按钮：重写“缩小”事件
        self.button_orange = QPushButton(self)
        self.button_orange.move(865, 11)
        self.button_orange.setFixedSize(18, 18)
        self.button_orange.setStyleSheet("QPushButton{background:orange;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
                                         "QPushButton:hover{background:#FFD306}"
                                         "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        # 绿色按钮：重写“最小化”事件
        self.button_green = QPushButton(self)
        self.button_green.move(830, 11)
        self.button_green.setFixedSize(18, 18)
        self.button_green.setStyleSheet("QPushButton{background:green;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
                                        "QPushButton:hover{background:#08BF14}"
                                        "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")

        # -------------------------------------------------
        #  实现界面图片轮播
        # -------------------------------------------------
        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(777, 333)
        self.classlabel.move(100, 222)
        self.classlabel.setPixmap(QPixmap("images/bg.jpg"))
        self.classlabel.setScaledContents(True)
        global lu
        self.n = 1
        self.lu = "./bg_images/" + str(self.n) + ".jpg"
        self.pm = QPixmap(self.lu)
        self.lbpic = myLabel(self)
        self.lbpic.setPixmap(self.pm)
        self.lbpic.resize(777, 333)
        self.lbpic.move(100, 222)
        self.lbpic.setScaledContents(True)
        self.lbpic._signal.connect(self.callbacklog)  # 连接信号
        self.timer1 = QTimer(self)
        self.timer1.timeout.connect(self.timer_TimeOut)
        self.timer1.start(2000)
        self.show()

    def timer_TimeOut(self):
        self.n += 1
        if self.n > 3:
            self.n = 1
        self.lu = "./bg_images/" + str(self.n) + ".jpg"
        self.pm = QPixmap(self.lu)
        self.lbpic.setPixmap(self.pm)

    def callbacklog(self, msg):
        from PIL import Image
        import matplotlib.pyplot as plt
        img = Image.open(self.lu)
        plt.figure("image")
        plt.imshow(img)
        plt.show()

    def quit_button(self):
        quit()

class myLabel(QLabel):
    _signal = pyqtSignal(str)
    def __init__(self, parent=None):
        super(myLabel, self).__init__(parent)

# -------------------------------------------------
#  个人信息界面定义
# -------------------------------------------------
class function2_view(QWidget):
    def __init__(self):
        super().__init__()

        self.classlabel = QLabel(self)
        self.classlabel.setText("基本信息")
        self.classlabel.move(22, 54)
        self.classlabel.setStyleSheet(
            "QLabel{color:#3A006F;font-size:24px;font-weight:bold;font-family:YaHei;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(911, 22)
        self.classlabel.move(21, 77)
        self.classlabel.setPixmap(QPixmap("images/fenge.png"))
        self.classlabel.setScaledContents(True)

        self.classlabel = QLabel(self)
        self.classlabel.setText("我的头像：")
        self.classlabel.move(37, 133)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )

        self.button_orange = QPushButton(self)
        self.button_orange.move(155, 108)
        self.button_orange.setFixedSize(66, 66)
        self.button_orange.setStyleSheet("QPushButton{border-radius: 33px}"
                                         "QPushButton{border-image: url(./images/touxiang.jpg)}")

        self.classlabel = QLabel(self)
        self.classlabel.setText("我的昵称：赵泽荣")
        self.classlabel.move(37, 204)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setText("  账号ID：20181611233")
        self.classlabel.move(37, 244)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setText("账号密码：********")
        self.classlabel.move(39, 284)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setText("  e-mail：729939433@qq.com")
        self.classlabel.move(35, 324)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setText("性别：")
        self.classlabel.move(435, 133)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )

        self.radioButton_1 = QtWidgets.QRadioButton(self)
        self.radioButton_1.setGeometry(QtCore.QRect(505, 133, 89, 16))
        self.radioButton_1.setStyleSheet("color:black;font-size:18px;font-weight:bold;font-family:KaiTi;")
        self.radioButton_1.setObjectName("radioButton_1")
        self.radioButton_2 = QtWidgets.QRadioButton(self)
        self.radioButton_2.setGeometry(QtCore.QRect(570, 133, 89, 16))
        self.radioButton_2.setStyleSheet("color:black;font-size:18px;font-weight:bold;font-family:KaiTi;")
        self.radioButton_2.setObjectName("radioButton_2")
        translate = QtCore.QCoreApplication.translate
        self.radioButton_1.setText(translate("Form", "男"))
        self.radioButton_1.setChecked(True)
        self.radioButton_2.setText(translate("Form", "女"))

        self.classlabel = QLabel(self)
        self.classlabel.setText("地区：")
        self.classlabel.move(435, 184)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )

        self.cb1 = QComboBox(self)
        self.cb1.move(503, 181)
        self.cb1.addItems(['中国大陆'])
        self.cb2 = QComboBox(self)
        self.cb2.move(606, 181)
        self.cb2.addItems(['重庆市', '山西省', '北京市'])
        self.cb3 = QComboBox(self)
        self.cb3.move(693, 181)
        self.cb3.addItems(['永川区', '晋中市', '大同市', '昌平区', '渝北区'])

        self.classlabel = QLabel(self)
        self.classlabel.setText("当前身份：用户精英")
        self.classlabel.move(398, 234)
        self.classlabel.setStyleSheet(
            "QLabel{color:#006030;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setText("当前等级：")
        self.classlabel.move(398, 284)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(30, 30)
        self.classlabel.move(500, 277)
        self.classlabel.setPixmap(QPixmap("images/jibie.png"))
        self.classlabel.setScaledContents(True)
        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(30, 30)
        self.classlabel.move(540, 277)
        self.classlabel.setPixmap(QPixmap("images/jibie.png"))
        self.classlabel.setScaledContents(True)
        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(30, 30)
        self.classlabel.move(580, 277)
        self.classlabel.setPixmap(QPixmap("images/jibie.png"))
        self.classlabel.setScaledContents(True)

        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(444, 1)
        self.classlabel.move(400, 333)
        self.classlabel.setPixmap(QPixmap("images/fenge2.jpg"))
        self.classlabel.setScaledContents(True)

        self.classlabel = QLabel(self)
        self.classlabel.setText(">> 更多设置")
        self.classlabel.move(747, 343)
        self.classlabel.setStyleSheet(
            "QLabel{color:blue;font-size:15px;font-weight:normal;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setText("服务公告")
        self.classlabel.move(22, 388)
        self.classlabel.setStyleSheet(
            "QLabel{color:#3A006F;font-size:23px;font-weight:bold;font-family:YaHei;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setText("☞ 站内公告站内公告站内公告站内公告站内公告站内公告")
        self.classlabel.move(37, 455)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )
        self.classlabel = QLabel(self)
        self.classlabel.setText("☞ 站内公告站内公告站内公告站内公告站内公告站内公告")
        self.classlabel.move(37, 485)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )
        self.classlabel = QLabel(self)
        self.classlabel.setText("☞ 站内公告站内公告站内公告站内公告站内公告站内公告")
        self.classlabel.move(37, 515)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )
        self.classlabel = QLabel(self)
        self.classlabel.setText("☞ 站内公告站内公告站内公告站内公告站内公告站内公告")
        self.classlabel.move(37, 545)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )
        self.classlabel = QLabel(self)
        self.classlabel.setText("☞ 站内公告站内公告站内公告站内公告站内公告站内公告")
        self.classlabel.move(37, 575)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setText("2021.09.25")
        self.classlabel.move(737, 455)
        self.classlabel.setStyleSheet(
            "QLabel{color:#7B7B7B;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )
        self.classlabel = QLabel(self)
        self.classlabel.setText("2021.09.25")
        self.classlabel.move(737, 485)
        self.classlabel.setStyleSheet(
            "QLabel{color:#7B7B7B;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )
        self.classlabel = QLabel(self)
        self.classlabel.setText("2021.09.25")
        self.classlabel.move(737, 515)
        self.classlabel.setStyleSheet(
            "QLabel{color:#7B7B7B;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )
        self.classlabel = QLabel(self)
        self.classlabel.setText("2021.09.25")
        self.classlabel.move(737, 545)
        self.classlabel.setStyleSheet(
            "QLabel{color:#7B7B7B;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )
        self.classlabel = QLabel(self)
        self.classlabel.setText("2021.09.25")
        self.classlabel.move(737, 575)
        self.classlabel.setStyleSheet(
            "QLabel{color:#7B7B7B;font-size:18px;font-weight:bold;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(911, 22)
        self.classlabel.move(21, 411)
        self.classlabel.setPixmap(QPixmap("images/fenge.png"))
        self.classlabel.setScaledContents(True)

        # -------------------------------------------------
        #  因关闭边框，模拟mac系统重写“最小化”、“缩小界面”、“关闭”按钮
        # -------------------------------------------------
        # 红色按钮：重写“关闭”事件
        self.button_red = QPushButton(self)
        self.button_red.move(900, 11)
        self.button_red.setFixedSize(18, 18)
        self.button_red.setStyleSheet(
            "QPushButton{background:#CE0000;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:red;}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        self.button_red.clicked.connect(self.quit_button)
        # 黄色按钮：重写“缩小”事件
        self.button_orange = QPushButton(self)
        self.button_orange.move(865, 11)
        self.button_orange.setFixedSize(18, 18)
        self.button_orange.setStyleSheet(
            "QPushButton{background:orange;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:#FFD306}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        # 绿色按钮：重写“最小化”事件
        self.button_green = QPushButton(self)
        self.button_green.move(830, 11)
        self.button_green.setFixedSize(18, 18)
        self.button_green.setStyleSheet(
            "QPushButton{background:green;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:#08BF14}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")

    def quit_button(self):
        quit()

# -------------------------------------------------
#  虫鼠识别界面定义
# -------------------------------------------------
class function3_view(QWidget):
    def __init__(self):
        super().__init__()

        # -------------------------------------------------
        #  因关闭边框，模拟mac系统重写“最小化”、“缩小界面”、“关闭”按钮
        # -------------------------------------------------
        # 红色按钮：重写“关闭”事件
        self.button_red = QPushButton(self)
        self.button_red.move(900, 11)
        self.button_red.setFixedSize(18, 18)
        self.button_red.setStyleSheet(
            "QPushButton{background:#CE0000;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:red;}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        self.button_red.clicked.connect(self.quit_button)
        # 黄色按钮：重写“缩小”事件
        self.button_orange = QPushButton(self)
        self.button_orange.move(865, 11)
        self.button_orange.setFixedSize(18, 18)
        self.button_orange.setStyleSheet(
            "QPushButton{background:orange;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:#FFD306}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        # 绿色按钮：重写“最小化”事件
        self.button_green = QPushButton(self)
        self.button_green.move(830, 11)
        self.button_green.setFixedSize(18, 18)
        self.button_green.setStyleSheet(
            "QPushButton{background:green;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:#08BF14}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")

        # -------------------------------------------------
        #  类别及概率信息显示
        # -------------------------------------------------
        self.classlabel = QLabel(self)
        self.classlabel.setText("类别：")
        self.classlabel.move(106, 522)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:21px;font-weight:bold;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setText("概率：")
        self.classlabel.move(106, 562)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:21px;font-weight:bold;font-family:KaiTi;}"
        )

        self.predclasslabel = QLabel(self)
        self.predclasslabel.setText(" 类别预测结果显示")
        self.predclasslabel.setFixedSize(211, 38)
        self.predclasslabel.move(169, 512)
        self.predclasslabel.setStyleSheet(
            "QLabel{background:#D2E9FF;border-width:3px;border-color:black}"
            "QLabel{color:rgb(300,300,300,120);font-size:21px;font-weight:bold;font-family:KaiTi;}"
        )
        self.predclasslabel.setWindowFlags(Qt.FramelessWindowHint)

        self.predvaluelabel = QLabel(self)
        self.predvaluelabel.setText(" 概率预测结果显示")
        self.predvaluelabel.setFixedSize(211, 38)
        self.predvaluelabel.move(169, 555)
        self.predvaluelabel.setStyleSheet(
            "QLabel{background:#D2E9FF;}"
            "QLabel{color:rgb(300,300,300,120);font-size:21px;font-weight:bold;font-family:KaiTi;}"
        )

        # -------------------------------------------------
        #  图片结果显示
        # -------------------------------------------------
        self.imglabel = QLabel(self)
        self.imglabel.setText("  待识别有害虫鼠图片显示")
        self.imglabel.setFixedSize(300, 300)
        self.imglabel.move(106, 198)
        self.imglabel.setStyleSheet(
            "QLabel{background:transparent;}"
            "QLabel{color:rgb(300,300,300,120);font-size:22px;font-weight:bold;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(329, 329)
        self.classlabel.move(92, 182)
        self.classlabel.setPixmap(QPixmap("./images/biankuang.png"))
        self.classlabel.setScaledContents(True)

        # -------------------------------------------------
        #  类别信息及灭除策略显示
        # -------------------------------------------------
        self.textlabel = QLabel(self)
        self.textlabel.setText("    类别信息及灭除策略显示")
        self.textlabel.setFixedSize(354, 399)
        self.textlabel.move(460, 198)
        self.textlabel.setStyleSheet(
            "QLabel{background:transparent;}"
            "QLabel{color:rgb(300,300,300,120);font-size:22px;font-weight:bold;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(373, 438)
        self.classlabel.move(449, 179)
        self.classlabel.setPixmap(QPixmap("./images/biankuang.png"))
        self.classlabel.setScaledContents(True)

        # -------------------------------------------------
        #  定义按钮事件
        # -------------------------------------------------
        # 打开图片按钮
        button_open_img = QPushButton(self)
        button_open_img.setText("打开图片")
        button_open_img.move(88, 99)
        button_open_img.setFixedSize(144, 66)
        button_open_img.clicked.connect(self.openimage)
        button_open_img.setStyleSheet(
            "QPushButton{background:#0080FF;color:white;font-size:19px;border-radius:24px;font-weight:bold;font-family: YaHei}"
            "QPushButton:hover{background:#2894FF}"
            "QPushButton:pressed{background:#004B97}")
        # 开始识别按钮
        button_detect_img = QPushButton(self)
        button_detect_img.setText("开始识别")
        button_detect_img.move(288, 99)
        button_detect_img.setFixedSize(144, 66)
        button_detect_img.clicked.connect(self.detectimage)
        button_detect_img.setStyleSheet(
            "QPushButton{background:#0080FF;color:white;font-size:18px;border-radius:24px;font-weight:bold;font-family: YaHei}"
            "QPushButton:hover{background:#2894FF}"
            "QPushButton:pressed{background:#004B97}")
        # 现实类别按钮
        button_detect_img = QPushButton(self)
        button_detect_img.setText("显示类别")
        button_detect_img.move(488, 99)
        button_detect_img.setFixedSize(144, 66)
        button_detect_img.clicked.connect(self.CategoryInformation)
        button_detect_img.setStyleSheet(
            "QPushButton{background:#0080FF;color:white;font-size:18px;border-radius:24px;font-weight:bold;font-family: YaHei}"
            "QPushButton:hover{background:#2894FF}"
            "QPushButton:pressed{background:#004B97}")
        # 灭除策略按钮
        button_detect_img = QPushButton(self)
        button_detect_img.setText("灭除策略")
        button_detect_img.move(688, 99)
        button_detect_img.setFixedSize(144, 66)
        button_detect_img.clicked.connect(self.EliminationStrategy)
        button_detect_img.setStyleSheet(
            "QPushButton{background:#0080FF;color:white;font-size:18px;border-radius:24px;font-weight:bold;font-family: YaHei}"
            "QPushButton:hover{background:#2894FF}"
            "QPushButton:pressed{background:#004B97}")

    def quit_button(self):
        quit()

    # -------------------------------------------------
    #  实现选择本地文件夹导出图片
    # -------------------------------------------------
    def openimage(self):
        self.imgName, self.imgType = QFileDialog.getOpenFileName(self, '选择图片', '.', '图像文件(*.jpg)')
        jpg_img = QtGui.QPixmap(self.imgName).scaled(self.imglabel.width(), self.imglabel.height())
        self.imglabel.setPixmap(jpg_img)

    # -------------------------------------------------
    #  实现虫鼠识别
    # -------------------------------------------------
    def detectimage(self):
        img_path = self.imgName
        pred_class, pred_value = detect(img_path)
        self.pd_class = pred_class
        pc = '  ' + str(pred_class)
        pv = '  ' + str(round(pred_value, 2)) + ' %'
        print(' 预测类别：', pred_class, '\n', '预测概率：', pred_value, ' %')
        self.predclasslabel.setText(pc)
        self.predvaluelabel.setText(pv)

    # -------------------------------------------------
    #  种类信息结果显示
    # -------------------------------------------------
    def CategoryInformation(self):
        pd_class_name1 = self.pd_class
        value1 = query(pd_class_name1)
        pd_value = str(value1)
        self.textlabel.setText(pd_value)
        self.textlabel.setWordWrap(True)
        self.textlabel.setStyleSheet(
            "QLabel{background:transparent;}"
            "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:KaiTi;}"
        )

    # -------------------------------------------------
    #  灭除策略显示
    # -------------------------------------------------
    def EliminationStrategy(self):
        pd_class_name2 = self.pd_class
        value2 = elimination_strategy(pd_class_name2)
        pd_value = str(value2)
        self.textlabel.setText(pd_value)
        self.textlabel.setWordWrap(True)
        self.textlabel.setStyleSheet(
            "QLabel{background:transparent;}"
            "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:KaiTi;}"
        )

    def quit_button(self):
        quit()

# -------------------------------------------------
#  定义检测结果界面
# -------------------------------------------------
class function4_view(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # -------------------------------------------------
        #  因关闭边框，模拟mac系统重写“最小化”、“缩小界面”、“关闭”按钮
        # -------------------------------------------------
        # 红色按钮：重写“关闭”事件
        self.button_red = QPushButton(self)
        self.button_red.move(900, 11)
        self.button_red.setFixedSize(18, 18)
        self.button_red.setStyleSheet(
            "QPushButton{background:#CE0000;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:red;}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        self.button_red.clicked.connect(self.quit_button)
        # 黄色按钮：重写“缩小”事件
        self.button_orange = QPushButton(self)
        self.button_orange.move(865, 11)
        self.button_orange.setFixedSize(18, 18)
        self.button_orange.setStyleSheet(
            "QPushButton{background:orange;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:#FFD306}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        # 绿色按钮：重写“最小化”事件
        self.button_green = QPushButton(self)
        self.button_green.move(830, 11)
        self.button_green.setFixedSize(18, 18)
        self.button_green.setStyleSheet(
            "QPushButton{background:green;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:#08BF14}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")

        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5_model/best.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='source')
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
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

        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = select_device(self.opt.device)
        self.half = 'cpu'

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
        self.pushButton_img.move(150, 100)
        self.pushButton_img.setFixedSize(188, 88)
        self.pushButton_img.setStyleSheet(
            "QPushButton{background:#0080FF;color:white;font-size:20px;border-radius:24px;font-weight:bold;font-family: YaHei}"
            "QPushButton:hover{background:#2894FF}"
            "QPushButton:pressed{background:#004B97}")
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_camera.move(400, 100)
        self.pushButton_camera.setFixedSize(188, 88)
        self.pushButton_camera.setStyleSheet(
            "QPushButton{background:#0080FF;color:white;font-size:20px;border-radius:24px;font-weight:bold;font-family: YaHei}"
            "QPushButton:hover{background:#2894FF}"
            "QPushButton:pressed{background:#004B97}")
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_video.move(650, 100)
        self.pushButton_video.setFixedSize(188, 88)
        self.pushButton_video.setStyleSheet(
            "QPushButton{background:#0080FF;color:white;font-size:20px;border-radius:24px;font-weight:bold;font-family: YaHei}"
            "QPushButton:hover{background:#2894FF}"
            "QPushButton:pressed{background:#004B97}")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.move(99, 211)
        self.label.setFixedSize(777, 411)
        self.label.setText("  待检测有害虫鼠图片显示")
        self.label.setStyleSheet(
            "QPushButton{background:#0080FF;color:white;font-size:20px;border-radius:24px;font-weight:bold;font-family: YaHei}"
            "QPushButton:hover{background:#2894FF}"
            "QPushButton:pressed{background:#004B97}")
        MainWindow.setCentralWidget(self.centralwidget)
        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(811, 433)
        self.classlabel.move(82, 199)
        self.classlabel.setPixmap(QPixmap("./images/biankuang.png"))
        self.classlabel.setScaledContents(True)
        self.classlabel = QLabel(self)
        self.classlabel.setText("注：本软件现仅支持检测老鼠")
        self.classlabel.setFixedSize(211, 45)
        self.classlabel.move(650, 565)
        self.classlabel.setStyleSheet(
            "QLabel{background:transparent;}"
            "QLabel{color:rgb(300,300,300,120);font-size:16px;font-family:楷书;}"
        )
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
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        name_list.append(self.names[int(cls)])
                        plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)

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
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            print(label)
                            plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)

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

# -------------------------------------------------
#  定义网页显示界面
# -------------------------------------------------
class function5_view(QWidget):
    def __init__(self):
        super().__init__()

        # -------------------------------------------------
        #  因关闭边框，模拟mac系统重写“最小化”、“缩小界面”、“关闭”按钮
        # -------------------------------------------------
        # 红色按钮：重写“关闭”事件
        self.button_red = QPushButton(self)
        self.button_red.move(900, 11)
        self.button_red.setFixedSize(18, 18)
        self.button_red.setStyleSheet(
            "QPushButton{background:#CE0000;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:red;}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        self.button_red.clicked.connect(self.quit_button)
        # 黄色按钮：重写“缩小”事件
        self.button_orange = QPushButton(self)
        self.button_orange.move(865, 11)
        self.button_orange.setFixedSize(18, 18)
        self.button_orange.setStyleSheet(
            "QPushButton{background:orange;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:#FFD306}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        # 绿色按钮：重写“最小化”事件
        self.button_green = QPushButton(self)
        self.button_green.move(830, 11)
        self.button_green.setFixedSize(18, 18)
        self.button_green.setStyleSheet(
            "QPushButton{background:green;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:#08BF14}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")

        # 新建一个QWebEngineView()对象
        self.qwebengine = QWebEngineView(self)
        # 设置网页在窗口中显示的位置和大小
        self.qwebengine.setGeometry(20, 50, 888, 555)
        # 在QWebEngineView中加载网址
        self.qwebengine.load(QUrl("http://www.chcdia.cn/"))

    def quit_button(self):
        quit()

# -------------------------------------------------
#  定义“与我联系”界面
# -------------------------------------------------
class function6_view(QWidget):
    def __init__(self):
        super().__init__()

        # -------------------------------------------------
        #  因关闭边框，模拟mac系统重写“最小化”、“缩小界面”、“关闭”按钮
        # -------------------------------------------------
        # 红色按钮：重写“关闭”事件
        self.button_red = QPushButton(self)
        self.button_red.move(900, 11)
        self.button_red.setFixedSize(18, 18)
        self.button_red.setStyleSheet(
            "QPushButton{background:#CE0000;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:red;}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        self.button_red.clicked.connect(self.quit_button)
        # 黄色按钮：重写“缩小”事件
        self.button_orange = QPushButton(self)
        self.button_orange.move(865, 11)
        self.button_orange.setFixedSize(18, 18)
        self.button_orange.setStyleSheet(
            "QPushButton{background:orange;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:#FFD306}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")
        # 绿色按钮：重写“最小化”事件
        self.button_green = QPushButton(self)
        self.button_green.move(830, 11)
        self.button_green.setFixedSize(18, 18)
        self.button_green.setStyleSheet(
            "QPushButton{background:green;color:white;box-shadow: 1px 1px 3px;border-radius: 9px}"
            "QPushButton:hover{background:#08BF14}"
            "QPushButton:pressed{border: 1px solid #3C3C3C!important;background:black}")

        self.classlabel = QLabel(self)
        self.classlabel.setText("联系我们")
        self.classlabel.move(400, 65)
        self.classlabel.setStyleSheet(
            "QLabel{color:#3A006F;font-size:32px;font-weight:bold;font-family:YaHei;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(888, 7)
        self.classlabel.move(22, 111)
        self.classlabel.setPixmap(QPixmap("images/fenge.png"))
        self.classlabel.setScaledContents(True)

        self.classlabel = QLabel(self)
        self.classlabel.setText("地址：重庆市永川区和顺大道大数据产业园C区")
        self.classlabel.move(44, 145)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:21px;font-weight:normal;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setText("电话：023-12345678  13293720189")
        self.classlabel.move(44, 185)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:21px;font-weight:normal;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setText("Email：729939433@qq.com")
        self.classlabel.move(44, 225)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:21px;font-weight:normal;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setText("QQ群：795347003")
        self.classlabel.move(44, 265)
        self.classlabel.setStyleSheet(
            "QLabel{color:black;font-size:21px;font-weight:normal;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setFixedSize(644, 273)
        self.classlabel.move(141, 313)
        self.classlabel.setPixmap(QPixmap("images/ditu.jpg"))
        self.classlabel.setScaledContents(True)

    def quit_button(self):
        quit()

# -------------------------------------------------
#  整合前面定义的界面，整体展示在系统界面中
# -------------------------------------------------
class StackedLayout(QMainWindow):
    def __init__(self):
        super(StackedLayout, self).__init__()

        self.setWindowFlags(Qt.FramelessWindowHint)  # 去边框
        self.resize(1131, 650)
        window_pale = QtGui.QPalette()
        window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap("images/1.jpg")))
        self.setPalette(window_pale)

        toolBar = QToolBar(self)
        self.addToolBar(Qt.LeftToolBarArea, toolBar)

        self.titlelabel = QLabel(self)
        self.titlelabel.setFixedSize(117, 117)
        self.titlelabel.move(35, 464)
        self.titlelabel.setPixmap(QPixmap("images/me.png"))
        self.titlelabel.setScaledContents(True)

        self.classlabel = QLabel(self)
        self.classlabel.setText("版本：V-2.0.1")
        self.classlabel.move(42, 577)
        self.classlabel.setStyleSheet(
            "QLabel{color:#424200;font-size:14px;font-weight:bold;font-family:KaiTi;}"
        )

        self.classlabel = QLabel(self)
        self.classlabel.setText("作者：赵泽荣")
        self.classlabel.move(48, 599)
        self.classlabel.setStyleSheet(
            "QLabel{color:#424200;font-size:14px;font-weight:bold;font-family:KaiTi;}"
        )

        functionHome = self.createButton0()
        functionHome.clicked.connect(lambda: self.onButtonClicked(0))
        functionHome.setFixedSize(177, 30)
        toolBar.addWidget(functionHome)
        function1 = self.createButton1('个人中心')
        function1.clicked.connect(lambda: self.onButtonClicked(1))
        function1.setFixedSize(177, 77)
        toolBar.addWidget(function1)
        function2 = self.createButton2('虫鼠识别')
        function2.clicked.connect(lambda: self.onButtonClicked(2))
        function2.setFixedSize(177, 77)
        toolBar.addWidget(function2)
        function3 = self.createButton3('虫鼠检测')
        function3.clicked.connect(lambda: self.onButtonClicked(3))
        function3.setFixedSize(177, 77)
        toolBar.addWidget(function3)
        function4 = self.createButton4('餐饮协会')
        function4.clicked.connect(lambda: self.onButtonClicked(4))
        function4.setFixedSize(177, 77)
        toolBar.addWidget(function4)
        function5 = self.createButton5('帮助联系')
        function5.clicked.connect(lambda: self.onButtonClicked(5))
        function5.setFixedSize(177, 77)
        toolBar.addWidget(function5)

        mainWidget = QWidget(self)
        self.mainLayout = QStackedLayout(mainWidget)
        self.mainLayout.addWidget(self.function1())
        self.mainLayout.addWidget(self.function2())
        self.mainLayout.addWidget(self.function3())
        self.mainLayout.addWidget(self.function4())
        self.mainLayout.addWidget(self.function5())
        self.mainLayout.addWidget(self.function6())
        mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(mainWidget)    # 设置中心窗口

    def onButtonClicked(self, index):
        if index < self.mainLayout.count():
            self.mainLayout.setCurrentIndex(index)

    def createButton0(self):
        btn = QToolButton(self)
        return btn

    def createButton1(self, text):
        btn = QToolButton(self)
        btn.setCheckable(True)
        btn.setText(text)
        icon = QIcon('images/home.png')
        btn.setIcon(icon)
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setAutoExclusive(True)
        return btn

    def createButton2(self, text):
        btn = QToolButton(self)
        btn.setCheckable(True)
        btn.setText(text)
        icon = QIcon('images/solve.png')
        btn.setIcon(icon)
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setAutoExclusive(True)
        return btn

    def createButton3(self, text):
        btn = QToolButton(self)
        btn.setCheckable(True)
        btn.setText(text)
        icon = QIcon('images/e.png')
        btn.setIcon(icon)
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setAutoExclusive(True)
        return btn

    def createButton4(self, text):
        btn = QToolButton(self)
        btn.setCheckable(True)
        btn.setText(text)
        icon = QIcon('images/add.png')
        btn.setIcon(icon)
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setAutoExclusive(True)
        return btn

    def createButton5(self, text):
        btn = QToolButton(self)
        btn.setCheckable(True)
        btn.setText(text)
        icon = QIcon('images/help.png')
        btn.setIcon(icon)
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setAutoExclusive(True)
        return btn

    def function1(self):
        self.classlabel = function1_view()
        return self.classlabel

    def function2(self):
        self.classlabel = function2_view()
        return self.classlabel

    def function3(self):
        self.classlabel = function3_view()
        return self.classlabel

    def function4(self):
        self.classlabel = function4_view()
        return self.classlabel

    def function5(self):
        self.classlabel = function5_view()
        return self.classlabel

    def function6(self):
        self.classlabel = function6_view()
        return self.classlabel

    # 重写鼠标移动事件
    def mouseMoveEvent(self, e: QMouseEvent):
        if self._tracking:
            self._endPos = e.pos() - self._startPos
            self.move(self.pos() + self._endPos)
            self.setCursor(QCursor(Qt.SizeAllCursor))
    # 重写鼠标“按”事件
    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._startPos = QPoint(e.x(), e.y())
            self._tracking = True
    # 重写鼠标“抬”事件
    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._tracking = False
            self._startPos = None
            self._endPos = None
            self.setCursor(QCursor(Qt.ArrowCursor))

if __name__ == '__main__':
    # -------------------------------------------------
    #  登录界面
    # -------------------------------------------------
    app1 = QApplication(sys.argv)
    mainWindow = LoginWindow()
    mainWindow.show()
    mainWindow.activateWindow()
    mainWindow.raise_()
    app1.exec_()
    # -------------------------------------------------
    #  系统界面
    # -------------------------------------------------
    app2 = QApplication(sys.argv)
    window = StackedLayout()
    window.show()
    sys.exit(app2.exec())

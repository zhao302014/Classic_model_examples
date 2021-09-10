#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class LoginWindow(QWidget):
    def __init__(self):
        super(LoginWindow, self).__init__()

        self.setWindowFlags(Qt.FramelessWindowHint)  # 去边框
        self.resize(1169, 731)
        window_pale = QtGui.QPalette()
        window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap("./background.jpg")))
        self.setPalette(window_pale)

        # title_pic
        self.titlelabel = QLabel(self)
        self.titlelabel.setFixedSize(130, 30)
        self.titlelabel.move(525, 167)
        self.titlelabel.setPixmap(QPixmap("./title.png"))
        self.titlelabel.setScaledContents(True)

        # logo_pic
        self.logolabel = QLabel(self)
        self.logolabel.setFixedSize(111, 99)
        self.logolabel.move(199, 50)
        self.logolabel.setPixmap(QPixmap("./logo_log.png"))
        self.logolabel.setScaledContents(True)

        # name_pic
        self.namelabel = QLabel(self)
        self.namelabel.setFixedSize(700, 120)
        self.namelabel.move(300, 40)
        self.namelabel.setPixmap(QPixmap("./name.png"))
        self.namelabel.setScaledContents(True)

        # 用户名
        self.le0 = QLineEdit(self)
        self.le0.move(440, 235)
        self.le0.resize(300, 38)
        self.le0.setPlaceholderText(" User Name")
        self.le0.setStyleSheet("background:transparent;border:0px;font-size:21px;color:white")
        self.le0.setClearButtonEnabled(True)

        # 密码
        self.le1 = QLineEdit(self)
        self.le1.move(440, 303)
        self.le1.resize(300, 38)
        self.le1.setPlaceholderText(" Password")
        self.le1.setStyleSheet("background:transparent;border:0px;font-size:21px;color:white")
        self.le1.setClearButtonEnabled(True)
        self.le1.setEchoMode(QLineEdit.Password)

        # 系统登录
        loginbtn = QPushButton(self)
        loginbtn.move(405, 424)
        loginbtn.resize(363, 45)
        loginbtn.clicked.connect(self.onLoginClick)
        loginbtn.setStyleSheet("background:transparent;border:0px")
        loginbtn.setCursor(QCursor(Qt.PointingHandCursor))

        # 账号注册
        register_btn = QPushButton(self)
        register_btn.move(400, 370)
        register_btn.setFixedSize(77, 33)
        register_btn.setStyleSheet("background:transparent;border:0px")
        register_btn.setCursor(QCursor(Qt.PointingHandCursor))
        register_btn.clicked.connect(self.register_btn)

        # 忘记密码
        forget_password_btn = QPushButton(self)
        forget_password_btn.move(700, 370)
        forget_password_btn.setFixedSize(77, 33)
        forget_password_btn.setStyleSheet("background:transparent;border:0px")
        forget_password_btn.setCursor(QCursor(Qt.PointingHandCursor))
        forget_password_btn.clicked.connect(self.forget_password)

        # 版权
        self.copyrightlabel = QLabel(self)
        self.copyrightlabel.setText("             Copyright @ 2021.09 · 赵泽荣 \n From Software College, Shanxi Agricultural University")
        self.copyrightlabel.move(340, 633)
        self.copyrightlabel.setStyleSheet(
            "QLabel{color:#E0E0E0;font-size:18px;}"
        )

    def register_btn(self):
        QMessageBox.information(self, "抱歉", "该功能正在抢修中...")

    def forget_password(self):
        QMessageBox.information(self, "抱歉", "该功能正在抢修中...")

    def onLoginClick(self):
        print('信息核验成功，正在进入系统...')
        qApp = QApplication.instance()
        qApp.quit()  # 关闭窗口

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
    app = QApplication(sys.argv)
    mainWindow = LoginWindow()
    mainWindow.show()
    mainWindow.activateWindow()
    mainWindow.raise_()
    app.exec_()

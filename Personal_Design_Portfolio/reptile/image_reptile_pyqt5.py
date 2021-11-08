#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.Qt import *
from PyQt5.QtWidgets import QProgressBar, QApplication, QLabel, QStatusBar, QPushButton
from image_reptile import reptile

class qt_view(QWidget):
    def __init__(self):
        super(qt_view, self).__init__()

        self.resize(445, 666)
        self.setWindowTitle("爬图百度")

        self.classlabel = QLabel(self)
        self.classlabel.setText("欢迎使用百度图片爬取APP")
        self.classlabel.move(90, 30)
        self.classlabel.setStyleSheet(
            "QLabel{color:blue;font-size:23px;font-weight:bold;font-family:YaHei;}"
        )

        window_pale = QtGui.QPalette()
        window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap("./background.jpg")))
        self.setPalette(window_pale)

        self.classlabel = QLabel(self)
        self.classlabel.setText("请填写您需要爬取的图片名：")
        self.classlabel.move(10, 100)
        self.classlabel.setStyleSheet(
            "QLabel{color:green;font-size:15px;font-weight:normal;font-family:YaHei;}"
        )

        self.le0 = QLineEdit(self)
        self.le0.move(15, 130)
        self.le0.resize(300, 28)
        self.le0.setPlaceholderText("eg：王者荣耀英雄图片")

        self.pathlabel = QLabel(self)
        self.pathlabel.setText("请填写您要存储图片的路径：")
        self.pathlabel.move(10, 175)
        self.pathlabel.setStyleSheet(
            "QLabel{color:green;font-size:15px;font-weight:normal;font-family:YaHei;}"
        )

        self.le1 = QLineEdit(self)
        self.le1.move(15, 205)
        self.le1.resize(300, 28)
        self.le1.setPlaceholderText("eg：F:/images")
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(6)
        le2btn = QPushButton(self)
        le2btn.setText("√")
        le2btn.setFont(font)
        le2btn.move(320, 129)
        le2btn.resize(30, 28)
        le2btn.clicked.connect(self.doAction)
        le1btn = QPushButton(self)
        le1btn.setText("...")
        le1btn.setFont(font)
        le1btn.move(320, 205)
        le1btn.resize(30, 28)
        le1btn.clicked.connect(self.select_folder)

        self.textlabel = QLabel(self)
        self.textlabel.setFixedSize(388, 352)
        self.textlabel.move(30, 252)
        self.textlabel.setPixmap(QPixmap("./images.jpg"))
        self.textlabel.setScaledContents(True)

        button_detect_img = QPushButton(self)
        button_detect_img.setText("start")
        button_detect_img.move(376, 130)
        button_detect_img.setFixedSize(55, 90)
        button_detect_img.clicked.connect(self.detectimage)

        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(15, 622, 430, 25)

        self.timer = QBasicTimer()
        self.step = 0

    def timerEvent(self, e):

        if self.step >= 100:
            self.step = 0
            self.pbar.setValue(self.step)
            self.timer.stop()
            return
        self.step = self.step+1
        self.pbar.setValue(self.step)

    def doAction(self, value):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(100, self)

    def break_value(self):
        if (self.le0.isModified()):
            print(self.le0.text())
        if (self.le1.isModified()):
            print(self.le1.text())

    def select_folder(self):
        self.directory = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "C:/")  # 起始路径
        print(self.directory)
        directory = str(self.directory)
        self.le1.setText(directory)

    def detectimage(self):
        if self.le0.text()=='' or self.le1.text()=='':
            QMessageBox.information(self, "提示", "请输入有效字符！")
        else:
            save_name = self.le0.text()
            save_path = self.le1.text()
            values = reptile(save_name, save_path)
            for value in range(60):
                print(next(values))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = qt_view()
    main.show()
    sys.exit(app.exec_())

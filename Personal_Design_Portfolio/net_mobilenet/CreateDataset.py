#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import random

train_ratio = 0.8  # 百分之80用来当训练集
test_ratio = 1 - train_ratio  # 百分之20用来当测试集

data_root_path = "../data/"  # 数据的根目录

train_list, test_list = [], []  # 创建一个训练列表和一个测试列表，用于读取里面每一类的类别
data_list = []

# 产生train.txt和test.txt
class_flag = -1
for a, b, c in os.walk(data_root_path):
    for i in range(len(c)):
        data_list.append(os.path.join(a, c[i]))

    for i in range(0, int(len(c) * train_ratio)):
        train_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
        train_list.append(train_data)

    for i in range(int(len(c) * train_ratio), len(c)):
        test_data = os.path.join(a, c[i]) + '\t' + str(class_flag) + '\n'
        test_list.append(test_data)

    class_flag += 1

print(train_list)
random.shuffle(train_list)  # 打乱次序
random.shuffle(test_list)   # 打乱次序

with open('mobilenet_train.txt', 'w', encoding='UTF-8') as f:
    for train_img in train_list:
        f.write(str(train_img))

with open('mobilenet_test.txt', 'w', encoding='UTF-8') as f:
    for test_img in test_list:
        f.write(test_img)

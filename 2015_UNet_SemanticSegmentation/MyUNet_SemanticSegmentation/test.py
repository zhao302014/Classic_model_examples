#!/usr/bin/python
# -*- coding:utf-8 -*-
# ------------------------------------------------- #
#      作者：赵泽荣
#      时间：2021年9月11日（农历八月初五）
#      个人站点：1.https://zhao302014.github.io/
#              2.https://blog.csdn.net/IT_charge/
#      个人GitHub地址：https://github.com/zhao302014
# ------------------------------------------------- #

import torch
import glob
import cv2
import numpy as np
from net import MyUNet

# 如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用 net 里定义的模型，如果 GPU 可用则将模型转到 GPU
model = MyUNet().to(device)
# 加载 train.py 里训练好的模型
model.load_state_dict(torch.load("./save_model/99model.pth"))

# 进入验证阶段
model.eval()

# 读取所有图片路径
tests_path = glob.glob('./data/test/*.png')
# 遍历素有图片
for test_path in tests_path:
    # 读取图片
    img = cv2.imread(test_path)
    # 转为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 转为 batch 为 1，通道为 1，大小为 img_shape*img_shape 的数组
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    # 转为tensor
    img_tensor = torch.from_numpy(img)
    # 将tensor拷贝到device中
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    # 预测
    pred = model(img_tensor)
    # 提取结果
    pred = np.array(pred.data.cpu()[0])[0]
    print(pred)
    # 处理结果
    pred[pred >= 0.99] = 255
    pred[pred < 0.99] = 0
    # 结果显示
    cv2.imshow('image', pred)
    cv2.waitKey(0)

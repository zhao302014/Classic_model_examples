#!/usr/bin/python
# -*- coding:utf-8 -*-
# ------------------------------------------------- #
#      作者：赵泽荣
#      时间：2021年9月26日（农历八月二十）
#      个人站点：1.https://zhao302014.github.io/
#              2.https://blog.csdn.net/IT_charge/
#      个人GitHub地址：https://github.com/zhao302014
# ------------------------------------------------- #
import cv2
import numpy as np
from utils.utils import ConvolutionalPoseMachine, draw_body_connections, draw_keypoints, draw_masks, draw_body_box

# 实例化 ConvolutionalPoseMachine 类（True 为使用预训练模型）
estimator = ConvolutionalPoseMachine(pretrained=True)
# opencv 读入图片
img = cv2.imread('data/image.jpg')
# 传入图片至实例化后的 ConvolutionalPoseMachine 类
pred_dict = estimator(img, masks=True, keypoints=True)
# 调用定义的 get_masks 静态方法获取掩膜
masks = estimator.get_masks(pred_dict['maskrcnn'], score_threshold=0.99)
# 调用定义的 get_keypoints 静态方法获取关键点
keypoints = estimator.get_keypoints(pred_dict['keypointrcnn'], score_threshold=0.99)
# 调用定义的 get_boxes 静态方法获取关键点
boxs = estimator.get_boxes(pred_dict['fasterrcnn'], score_threshold=0.99)
# BGR转灰度图像
image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 合并单通道成多通道
image_dst = cv2.merge([image_gray] * 3)
# 绘制掩膜
result_m = draw_masks(image_dst, masks, color=(0, 255, 0), alpha=0.5)
# 绘制预测框
result_b = draw_body_box(img, boxs, thickness=3)
# 连接关键点
result_k = draw_body_connections(img, keypoints, thickness=4, alpha=0.7)
# 绘制关键点
result_k = draw_keypoints(result_k, keypoints, radius=5, alpha=0.8)
# 全部绘制在一张图上
result1 = draw_body_box(result_m, boxs, thickness=3)
result = draw_body_connections(result1, keypoints, thickness=4, alpha=0.7)
result = draw_keypoints(result, keypoints, radius=5, alpha=0.8)
# 将参数元组的元素数组按水平方向及垂直方向进行叠加
# 预计显示结果如下示意：
#      —————————————————————————————————
#     |       原图       |   掩膜预测图   |
#      —————————————————————————————————
#     | 关键点及连接绘制图  |  预测框绘制图  |
#      —————————————————————————————————
# 水平排列
image_h1 = np.hstack((img, result_m))
image_h2 = np.hstack((result_k, result_b))
# 垂直排列
image_v_and_h = np.vstack((image_h1, image_h2))
# 展现单图结果
cv2.imshow('Image Separate Show', image_v_and_h)
cv2.imshow('Image Total Show', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

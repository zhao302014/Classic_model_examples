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
# opencv 读入视频
cap = cv2.VideoCapture('data/video.mp4')

# 读取成功意味着 cap.isOpened()==True，持续运行
while True:
    # frame 相当于一帧一帧的图像
    _, frame = cap.read()
    # 传入视频帧至实例化后的 ConvolutionalPoseMachine 类
    pred_dict = estimator(frame, masks=True, keypoints=True)
    # 调用定义的 get_masks 静态方法获取掩膜
    masks = estimator.get_masks(pred_dict['maskrcnn'], score_threshold=0.99)
    # 调用定义的 get_keypoints 静态方法获取关键点
    keypoints = estimator.get_keypoints(pred_dict['keypointrcnn'], score_threshold=0.99)
    # 调用定义的 get_boxes 静态方法获取关键点
    boxs = estimator.get_boxes(pred_dict['fasterrcnn'], score_threshold=0.99)
    # BGR 转灰度图像
    frame_dst = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 合并单通道成多通道
    frame_dst = cv2.merge([frame_dst] * 3)
    # 绘制掩膜
    overlay_m = draw_masks(frame_dst, masks, color=(0, 255, 0), alpha=0.5)
    # 绘制预测框
    overlay_b = draw_body_box(frame_dst, boxs, thickness=3)
    # 连接关键点
    overlay_k = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
    # 绘制关键点
    overlay_k = draw_keypoints(overlay_k, keypoints, radius=4, alpha=0.8)
    # 将参数元组的元素数组按水平方向及垂直方向进行叠加
    # 预计显示结果如下示意：
    #      —————————————————————————————————
    #     |       原图       |   掩膜预测图   |
    #      —————————————————————————————————
    #     | 关键点及连接绘制图  |  预测框绘制图  |
    #      —————————————————————————————————
    # 水平排列
    image_h1 = np.hstack((frame, overlay_m))
    image_h2 = np.hstack((overlay_k, overlay_b))
    # 垂直排列
    image_v_and_h = np.vstack((image_h1, image_h2))
    # 处理后的视频帧显示
    cv2.imshow('Video Show', image_v_and_h)
    # cv2.waitKey(x)：x数值越小，理论上运行越快（运行速度也与电脑硬件运行处理图片速度有关）
    if cv2.waitKey(1) & 0xff == 27:  # exit if pressed `ESC`（或按ESC退出）
        break
# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()

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
import torch
import numpy as np
import torchvision

'''
  创建一个“人体姿态估计器”类
'''
class ConvolutionalPoseMachine(object):
    def __init__(self, pretrained=False):
        # 是否使用 maskrcnn_resnet50_fpn 预训练模型，true 为使用，false 为不使用，默认为 false
        self._maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
        # 是否使用 keypointrcnn_resnet50_fpn 预训练模型，true 为使用，false 为不使用，默认为 false
        self._keypointrcnn = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=pretrained)
        # 是否使用 fasterrcnn_resnet50_fpn 预训练模型，true 为使用，false 为不使用，默认为 false
        self._fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        # 如果 GPU 存在，则转为 cuda 运行
        if torch.cuda.is_available():
            self._maskrcnn = self._maskrcnn.cuda()
            self._keypointrcnn = self._keypointrcnn.cuda()
            self._fasterrcnn = self._fasterrcnn.cuda()
        # 将模型转为验证模式
        self._maskrcnn.eval()
        self._keypointrcnn.eval()
        self._fasterrcnn.eval()

    def __call__(self, image, masks=True, keypoints=True, boxs=True):
        # 调用下面定义的 _transform_image 方法传入 image，将 image 转为 tensor 格式
        x = self._transform_image(image)
        # 如果 GPU 存在，则转为 cuda 运行
        if torch.cuda.is_available():
            x = x.cuda()
        # 若 masks 为 True 则进行掩膜操作，否则不执行此语句，默认为 True
        m = self._predict_masks(x) if masks else [None]
        # 若 keypoints 为 True 则进行关键点检测操作，否则不执行此语句，默认为 True
        k = self._predict_keypoints(x) if keypoints else [None]
        # 若 boxes 为 True 则进行关键点检测操作，否则不执行此语句，默认为 True
        b = self._predict_boxes(x) if boxs else [None]
        # 以 “键值对” 形式返回掩膜及关键点检测结果（注：m、k、b 为列表，要获取的是列表里的值，故 “[0]”）
        return {'maskrcnn': m[0], 'keypointrcnn': k[0], 'fasterrcnn': b[0]}

    # 定义转换 image 格式函数
    def _transform_image(self, image):
        # 返回值：将图像由 numpy 格式转为 tensor 格式
        return torchvision.transforms.ToTensor()(image)

    # 定义掩膜预测函数
    def _predict_masks(self, x):
        # 被包含部分不进行梯度计算
        with torch.no_grad():
            # 返回值：将 tensor 格式 image 传入掩膜预测模型
            return self._maskrcnn([x])

    # 定义关键点预测函数
    def _predict_keypoints(self, x):
        # 被包含部分不进行梯度计算
        with torch.no_grad():
            # 返回值：将 tensor 格式 image 传入关键点预测模型
            return self._keypointrcnn([x])

    # 定义预测框预测函数
    def _predict_boxes(self, x):
        # 被包含部分不进行梯度计算
        with torch.no_grad():
            # 返回值：将 tensor 格式 image 传入关键点预测模型
            return self._fasterrcnn([x])

    # 静态方法 类或实例均可调用
    @staticmethod
    def get_masks(dictionary, label=1, score_threshold=0.5):
        # 定义一个空掩膜列表
        masks = []
        # 此处的 dictionary 相当于前面返回值中的 m[0]
        if dictionary:
            # 碾平后，依次循环 非零 且 与 label 相等的 dictionary 中 labels 标签
            for i in (dictionary['labels'] == label).nonzero().view(-1):
                # 若标签对应的 scores 值(置信度)大于预先设定的阈值，则将掩膜存入列表中
                if dictionary['scores'][i] > score_threshold:
                    # 若标签对应的 masks 值大于 0.5，则将 true 传入 mask，否则传入 false（即大于 0.5 显示掩膜）
                    mask = dictionary['masks'][i].detach().cpu().squeeze().numpy() > 0.5
                    # 将掩膜存入列表中
                    masks.append(mask)
        # return 值：将 masks 转为矩阵格式返回
        return np.asarray(masks, dtype=np.uint8)

    @staticmethod
    def get_keypoints(dictionary, label=1, score_threshold=0.5):
        keypoints = []
        # 此处的 dictionary 相当于前面返回值中的 k[0]
        if dictionary:
            for i in (dictionary['labels'] == label).nonzero().view(-1):
                if dictionary['scores'][i] > score_threshold:
                    keypoint = dictionary['keypoints'][i].detach().cpu().squeeze().numpy()
                    keypoints.append(keypoint)
        return np.asarray(keypoints, dtype=np.int32)

    @staticmethod
    def get_boxes(dictionary, label=1, score_threshold=0.5):
        boxes = []
        # 此处的 dictionary 相当于前面返回值中的 b[0]
        if dictionary:
            for i in (dictionary['labels'] == label).nonzero().view(-1):
                if dictionary['scores'][i] > score_threshold:
                    box = dictionary['boxes'][i].detach().cpu().squeeze().numpy()
                    boxes.append(box)
        return np.asarray(boxes, dtype=np.int32)

'''
  定义一系列绘制掩膜、绘制关键点、连接关键点、绘制预测框的函数
'''
# 定义一个掩膜颜色定义函数
def _colorize_mask(mask, color=None):
    # 没有传入颜色则随机产生颜色，若传入颜色则按传入颜色来绘制
    b = mask * np.random.randint(0, 255) if not color else mask * color[0]
    g = mask * np.random.randint(0, 255) if not color else mask * color[1]
    r = mask * np.random.randint(0, 255) if not color else mask * color[2]
    # 返回值：R、G、B 三通道合并后的图
    return cv2.merge((b, g, r))

# 定义一个关键点绘制函数
def _draw_keypoint(image, point, color, radius=1):
    # point返回值是包含三个数字的列表，分别表示横、纵坐标及点半径
    x, y, r = point
    if int(r):
        # 用原点形式绘制关键点（注：cv2.LINE_AA 为抗锯齿,这样看起来会非常平滑）
        cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    return image

# 定义一个关键点连接函数
def _draw_connection(image, point1, point2, color, thickness=1):
    x1, y1, v1 = point1
    x2, y2, v2 = point2
    if int(v1) and int(v2):
        # 连接关键点用直线（注：cv2.LINE_AA 为抗锯齿,这样看起来会非常平滑）
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
    return image

# 定义一个预测框预测函数
def _draw_box(image, point1, point2, point3, point4, color, thickness=1):
    # point1 ~ point4 代表框选人物矩形的四个点位置（注：yolo 中也是用同样的定点法）
    cv2.rectangle(image, (int(point1), int(point2)), (int(point3), int(point4)), color, thickness, cv2.LINE_AA)
    return image

# 绘制掩膜
def draw_masks(image, masks, color=None, alpha=0.5):
    # 在拷贝的image附件中执行下述代码
    result = image.copy()
    for mask in masks:
        # 显示图片前必须先转为uint8格式
        mask_bin = np.uint8(mask > 0)
        # 通道融合
        mask_inv = cv2.merge([1 - mask_bin] * 3)
        # 绘制关键语句，调用前面定义的_colorize_mask函数
        mask_rgb = _colorize_mask(mask_bin, color)
        # 彩色图像数组和掩膜图像数组相乘
        result = cv2.multiply(result, mask_inv)
        # 彩色图像数组和掩膜图像数组相加
        result = cv2.add(result, mask_rgb)
    # 返回值：将原图像与掩膜叠加
    return cv2.addWeighted(result, alpha, image, 1.0 - alpha, 0)

# 绘制关键点
def draw_keypoints(image, keypoints, radius=1, alpha=1.0):
    result = image.copy()
    for kp in keypoints:
        for p in kp:
            # 绘制关键语句，调用前面定义的_draw_keypoint函数
            result = _draw_keypoint(result, p, (0, 255, 0), radius)
    return cv2.addWeighted(result, alpha, image, 1.0 - alpha, 0)

# 连接关键点
def draw_body_connections(image, keypoints, thickness=1, alpha=1.0):
    result = image.copy()
    b_conn = [(0, 5), (0, 6), (5, 6), (5, 11), (6, 12), (11, 12)]
    h_conn = [(0, 1), (0, 2), (1, 3), (2, 4)]
    l_conn = [(5, 7), (7, 9), (11, 13), (13, 15)]
    r_conn = [(6, 8), (8, 10), (12, 14), (14, 16)]
    for kp in keypoints:
        for i, j in b_conn:
            result = _draw_connection(result, kp[i], kp[j], (0, 255, 255), thickness)
        for i, j in h_conn:
            result = _draw_connection(result, kp[i], kp[j], (0, 255, 255), thickness)
        for i, j in l_conn:
            result = _draw_connection(result, kp[i], kp[j], (255, 255, 0), thickness)
        for i, j in r_conn:
            result = _draw_connection(result, kp[i], kp[j], (255, 0, 255), thickness)
    return cv2.addWeighted(result, alpha, image, 1.0 - alpha, 0)

# 绘制预测框
def draw_body_box(image, keypoints, thickness=1):
    result = image.copy()
    for kp in keypoints:
        result = _draw_box(result, kp[0], kp[1], kp[2], kp[3], (0, 255, 255), thickness)
    return cv2.addWeighted(result, 0.5, image, 1.0 - 0.5, 0)

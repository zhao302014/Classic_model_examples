#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import cv2
from utils.plots import plot_one_box
from detector import Get_Pred

# 摄像头检测
def detect_cap():
    capture = cv2.VideoCapture(0)
    while True:
        ret, img = capture.read()
        if ret is not True:
            break
        # 实例化模型推理类
        detect = Get_Pred(model_path)
        # 获取json格式下的模型预测输出
        value_json = detect.process(img)
        # 解码json格式数据（得到字典格式数据）
        value = json.loads(value_json)

        # 打印输出预测结果
        pred_num = value['total num']
        pred_xyxy = value['xyxy']
        print('\n')
        print('数量：', pred_num)
        print('预测框位置坐标：', pred_xyxy)
        for xyxy in pred_xyxy:
            print(xyxy)
            plot_one_box(xyxy, img, label=None, color=(255, 255, 0), line_thickness=3)

        num = 'num:' + str(pred_num)
        cv2.putText(img, str(num), (50, 150), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 视频检测
def detect_video(path):
    capture = cv2.VideoCapture(path)
    while True:
        ret, img = capture.read()
        if ret is not True:
            break
        # img = cv2.imread(image_path)
        # 实例化模型推理类
        detect = Get_Pred(model_path)
        # 获取json格式下的模型预测输出
        value_json = detect.process(img)
        # 解码json格式数据（得到字典格式数据）
        value = json.loads(value_json)

        # 打印输出预测结果
        pred_num = value['total num']
        pred_xyxy = value['xyxy']
        print('\n')
        print('数量：', pred_num)
        print('预测框位置坐标：', pred_xyxy)
        for xyxy in pred_xyxy:
            print(xyxy)
            plot_one_box(xyxy, img, label=None, color=(255, 255, 0), line_thickness=3)

        num = 'num:' + str(pred_num)
        cv2.putText(img, str(num), (50, 150), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 图像检测
def detect_img(path):
    img = cv2.imread(path)
    # 实例化模型推理类
    detect = Get_Pred(model_path)
    # 获取json格式下的模型预测输出
    value_json = detect.process(img)
    # 解码json格式数据（得到字典格式数据）
    value = json.loads(value_json)

    # 打印输出预测结果
    pred_num = value['total num']
    pred_xyxy = value['xyxy']
    print('\n')
    print('数量：', pred_num)
    print('预测框位置坐标：', pred_xyxy)
    for xyxy in pred_xyxy:
        print(xyxy)
        plot_one_box(xyxy, img, label=None, color=(255, 255, 0), line_thickness=3)

    num = 'num:' + str(pred_num)
    cv2.putText(img, str(num), (50, 150), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 定义模型路径
    model_path = 'weights/yolov5s.pt'

    print('请选择使用何种方式进行检测（键入数字即可）：' + '\n' + '   [1] -> 图片'+ '\n' + '   [2] -> 视频'+ '\n' + '   [3] -> 摄像头')
    key_input_num = input('\n' + '您的选择是：')

    # 选择使用何种方式进行预测
    if key_input_num == '1':
        key_input_path = input('输入待检测图片路径：')
        detect_img(key_input_path)
    elif key_input_num == '2':
        key_input_path = input('输入待检测视频路径：')
        detect_video(key_input_path)
    elif key_input_num == '3':
        detect_cap()
    else:
        print('输入有误，请重启程序重新输入！')



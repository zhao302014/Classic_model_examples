#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import cv2
from utils.plots import plot_one_box
from detector import Get_Pred

# 计算 IoU
def cal_iou(pxmin, pymin, pxmax, pymax, gxmin, gymin, gxmax, gymax):
    parea = (pxmax - pxmin) * (pymax - pymin)
    garea = (gxmax - gxmin) * (gymax - gymin)
    xmin = max(pxmin, gxmin)
    ymin = max(pymin, gymin)
    xmax = min(pxmax, gxmax)
    ymax = min(pymax, gymax)
    w = xmax - xmin
    h = ymax - ymin
    if w < 0 or h < 0:
        return 0
    area = w * h
    return area / (parea + garea - area)

# 获取鼠标位置
def draw_circle(event, x, y, flags, param):
    global xmin, ymin, xmax, ymax
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        xmin = x
        ymin = y
    elif event == cv2.EVENT_LBUTTONUP:
        print(x, y)
        xmax = x
        ymax = y

# 视频检测
def detect_video(path):
    global xmin, ymin, xmax, ymax
    pred_list = []
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', draw_circle)

    capture = cv2.VideoCapture(path)

    while True:
        ret, img = capture.read()
        img_copy = img.copy()
        k = cv2.waitKey(1)
        if ret is not True:
            break

        detect = Get_Pred(model_path)
        value_json = detect.process(img)
        value = json.loads(value_json)

        pred_xyxy = value['xyxy']
        for xyxy in pred_xyxy:
            plot_one_box(xyxy, img, label=None, color=(255, 255, 0), line_thickness=1)
            print(xmin, ymin, xmax, ymax)
            area = cal_iou(xyxy[0], xyxy[1], xyxy[2], xyxy[3], xmin, ymin, xmax, ymax)
            if area >= 0.25:
                xyxy_new = [xyxy[0] - 2, xyxy[1] - 2, xyxy[2] - 2, xyxy[3] - 2]
                plot_one_box(xyxy_new, img, label=None, color=(0, 255, 255), line_thickness=1)
                pred_point = (int((xyxy_new[2] + xyxy_new[0]) / 2), int((xyxy_new[3] + xyxy_new[1]) / 2))
                cv2.circle(img, pred_point, 1, (255, 255, 255), 1)
                pred_list.append(pred_point)
                for save_content in pred_list:
                    cv2.circle(img, save_content, 1, (0, 0, 255), 1)
                xmin, ymin, xmax, ymax = xyxy_new[0], xyxy_new[1], xyxy_new[2], xyxy_new[3]
            if k == 32:
                cv2.imshow('img', img_copy)
                try:
                    cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                except:
                    pass
                cv2.imshow('img', img_copy)
                cv2.waitKey(0)

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    model_path = 'weights/yolov5s.pt'
    pred_path = 'images/demo.mp4'

    xmin, ymin, xmax, ymax = 0, 0, 0, 0

    detect_video(pred_path)

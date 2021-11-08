#!/usr/bin/python
# -*- coding:utf-8 -*-
from PIL import Image
import os
import glob

def resizePic(oldPic, newPic, width=256, height=256):
    img = Image.open(oldPic)      # 打开resize前的图片（PIL类型）
    try:
        new_img = img.resize((width, height), Image.BILINEAR)    # resize图片大小到指定width*height（双线性插值法）
        new_img.save(newPic)      # 保存resize后的图片
    except Exception as e:
        print(e)

if __name__ == '__main__':
    parent_folder_path = '../data/'   # 父文件夹path
    sub_folder_name = os.listdir(parent_folder_path)   # 子文件夹name集合
    for sub_images_path in sub_folder_name:  # 遍历子文件夹name
        new_sub_images_path = os.path.join(parent_folder_path, sub_images_path)  # 路径拼接，索引到子文件夹
        for sub_jpgPic_file in glob.glob(new_sub_images_path + r"\\*.jpg"):      # 获取指定目录下的所有图片
            print(sub_jpgPic_file)
            resizePic(sub_jpgPic_file, sub_jpgPic_file)
            print(sub_jpgPic_file)

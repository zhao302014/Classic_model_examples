#!/usr/bin/python
# -*- coding:utf-8 -*-
import os

parent_folder_path = '../data'                      # 父文件夹path
sub_folder_name = os.listdir(parent_folder_path)    # 子文件夹name集合

for sub_images_path in sub_folder_name:             # 遍历子文件夹name
    new_sub_images_path = os.path.join(parent_folder_path, sub_images_path)     # 路径拼接，索引到子文件夹
    new_sub_images_name = os.listdir(new_sub_images_path)       # 子文件夹下图片name集合
    i = 1
    for new_sub_image_name in new_sub_images_name:  # 遍历子文件夹下图片name
        new_sub_image_path = os.path.join(new_sub_images_path, new_sub_image_name)  # 路径拼接，索引到子文件夹下的每张图片
        old_name = new_sub_image_path            # 旧名字
        new_name = os.path.join(new_sub_images_path + '/' + sub_images_path + '_' + str(i) + '.jpg')   # 新名字
        os.rename(old_name, new_name)            # 重命名
        print("旧路径：", old_name, " --->  重命名后路径：", new_name)
        i += 1

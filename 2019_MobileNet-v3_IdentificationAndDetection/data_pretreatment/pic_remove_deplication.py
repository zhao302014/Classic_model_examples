#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import hashlib

# 获取文件的md5
def get_md5(file):
    file = open(file, 'rb')
    md5 = hashlib.md5(file.read())
    file.close()
    md5_values = md5.hexdigest()
    return md5_values

# 单文件夹去重
def remove_by_md5_singledir(file_dir):
    file_list = os.listdir(file_dir)
    md5_list = []
    print("去重前图像数量：" + str(len(file_list)))
    for filepath in file_list:
        filemd5 = get_md5(os.path.join(file_dir, filepath))
        if filemd5 not in md5_list:
            md5_list.append(filemd5)
        else:
            os.remove(os.path.join(file_dir, filepath))
    print("去重后图像数量：" + str(len(os.listdir(file_dir))))

if __name__ == '__main__':
    file_dir = '../data/***/'
    remove_by_md5_singledir(file_dir)

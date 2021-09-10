#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np

# 数据增强之缩放操作
def Scale(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

# 数据增强之水平翻转
def Flip_Horizontal(image):
    return cv2.flip(image, 1, dst=None)  # 水平镜像

# 数据增强之垂直翻转
def Flip_Vertical(image):
    return cv2.flip(image, 0, dst=None)  # 垂直镜像

# 数据增强之旋转
def Rotate(image, angle=15, scale=0.9):
    w = image.shape[1]
    h = image.shape[0]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)  # 旋转矩阵
    image = cv2.warpAffine(image, M, (w, h))   # 旋转
    return image

# 数据增强之变暗
def Darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy

# 数据增强之变亮
def Brighter(image, percetage=1.1):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy

# 数据增强之平移
def Translation(img, x, y):
    img_info = img.shape
    height = img_info[0]
    width = img_info[1]
    mat_translation = np.float32([[1, 0, x], [0, 1, y]])  # 变换矩阵：设置平移变换所需的计算矩阵：2行3列（平移变换：其中x表示水平方向上的平移距离，y表示竖直方向上的平移距离）
    dst = cv2.warpAffine(img, mat_translation, (width, height))  # 变换函数
    return dst

# 数据增强之增加椒盐噪声
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg

# 数据增强之增加高斯噪声
def GaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg

# 数据增强之增加高斯滤波
def Blur(img):
    blur = cv2.GaussianBlur(img, (7, 7), 1.5)   # cv2.GaussianBlur(图像，卷积核，标准差）
    return blur

def Data_Augmentation(rootpath):
    for sub_path, sub_name_list, pic_name_list in os.walk(rootpath):   # 遍历父文件夹，一次返回子文件夹路径，子文件夹名字列表，子文件夹下图片名字列表
        for pic_name in pic_name_list:
            sub_pic_path = os.path.join(sub_path, pic_name)       # ./data/fly\fly_1.jpg
            split = os.path.split(sub_pic_path)                   # ('./data/fly', 'fly_1.jpg')
            dir_loc = os.path.split(split[0])[1]                  # split[0]：./data/fly；  split：('./data', 'pillworm')；  [1]：fly
            save_path = os.path.join(rootpath, dir_loc)           # ./data/fly

            sub_pic = cv2.imread(sub_pic_path)

            img_scale = Scale(sub_pic, 1.5)      # 数据增强之缩放操作
            img_scale_path = os.path.join(save_path, pic_name[:-4] + "_scale.jpg")
            cv2.imwrite(img_scale_path, img_scale)
            print(img_scale_path + '：Data Augmentation Scale Success!')

            img_flip_horizontal = Flip_Horizontal(sub_pic)    # 数据增强之水平翻转
            img_flip_horizontal_path = os.path.join(save_path, pic_name[:-4] + "_horizontal.jpg")
            cv2.imwrite(img_flip_horizontal_path, img_flip_horizontal)
            print(img_flip_horizontal_path + '：Data Augmentation Flip_Horizontal Success!')

            img_flip_vertical = Flip_Vertical(sub_pic)        # 数据增强之垂直翻转
            img_flip_vertical_path = os.path.join(save_path, pic_name[:-4] + "_vertical.jpg")
            cv2.imwrite(img_flip_vertical_path, img_flip_vertical)
            print(img_flip_vertical_path + '：Data Augmentation Flip_Vertical Success!')

            img_rotate_90 = Rotate(sub_pic, 90)             # 数据增强之旋转
            img_rotate_90_path = os.path.join(save_path, pic_name[:-4] + "_rotate90.jpg")
            cv2.imwrite(img_rotate_90_path, img_rotate_90)
            print(img_rotate_90_path + '：Data Augmentation Rotate_90 Success!')

            img_rotate_180 = Rotate(sub_pic, 180)            # 数据增强之旋转
            img_rotate_180_path = os.path.join(save_path, pic_name[:-4] + "_rotate180.jpg")
            cv2.imwrite(img_rotate_180_path, img_rotate_180)
            print(img_rotate_180_path + '：Data Augmentation Rotate_180 Success!')

            img_rotate_270 = Rotate(sub_pic, 270)            # 数据增强之旋转
            img_rotate_270_path = os.path.join(save_path, pic_name[:-4] + "_rotate270.jpg")
            cv2.imwrite(img_rotate_270_path, img_rotate_270)
            print(img_rotate_270_path + '：Data Augmentation Rotate_270 Success!')

            img_translation = Translation(sub_pic, 15, 15)      # 数据增强之平移
            img_translation_path = os.path.join(save_path, pic_name[:-4] + "_move.jpg")
            cv2.imwrite(img_translation_path, img_translation)
            print(img_translation_path + '：Data Augmentation Translation Success!')

            img_darker = Darker(sub_pic)                 # 数据增强之变暗
            img_darker_path = os.path.join(save_path, pic_name[:-4] + "_darker.jpg")
            cv2.imwrite(img_darker_path, img_darker)
            print(img_darker_path + '：Data Augmentation Darker Success!')

            img_brighter = Brighter(sub_pic)             # 数据增强之变亮
            img_brighter_path = os.path.join(save_path, pic_name[:-4] + "_brighter.jpg")
            cv2.imwrite(img_brighter_path, img_brighter)
            print(img_brighter_path + '：Data Augmentation Brighter Success!')

            img_blur = Blur(sub_pic)                     # 数据增强之增加高斯滤波
            img_blur_path = os.path.join(save_path, pic_name[:-4] + "_blur.jpg")
            cv2.imwrite(img_blur_path, img_blur)
            print(img_blur_path + '：Data Augmentation Blur Success!')

            img_salt = SaltAndPepper(sub_pic, 0.05)      # 数据增强之增加椒盐噪声
            img_salt_path = os.path.join(save_path, pic_name[:-4] + "_salt.jpg")
            cv2.imwrite(img_salt_path, img_salt)
            print(img_salt_path + '：Data Augmentation SaltAndPepper Success!')

            img_gaussian = GaussianNoise(sub_pic, 0.05)  # 数据增强之增加高斯噪声
            img_gaussian_path = os.path.join(save_path, pic_name[:-4] + "_gaussian.jpg")
            cv2.imwrite(img_gaussian_path, img_gaussian)
            print(img_gaussian_path + '：Data Augmentation GaussianNoise Success!')

if __name__ == "__main__":
    root_path = "../data/"
    Data_Augmentation(root_path)
    print('Done!')

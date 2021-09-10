#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Data_Loader(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag
        self.targetsize = 224     # 把图片压缩成224X224

        # 训练集的处理方法
        self.train_tf = transforms.Compose([
                transforms.Resize(self.targetsize),   # 压缩图片
                transforms.RandomHorizontalFlip(),    # 随机水平反转
                transforms.RandomVerticalFlip(),      # 随机垂直反转图片
                transforms.ToTensor(),                # 把图片转变为Tensor()格式
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   # 图片标准化
            ])

        #  验证集（测试集）的处理方法
        self.val_tf = transforms.Compose([
                transforms.Resize(self.targetsize),    # 压缩图片
                transforms.ToTensor(),                 # 把图片转变为Tensor()格式
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 图片标准化
            ])

    # 通过读取txt文档内容，返回文档中的每一条信息
    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        return imgs_info

    def padding_black(self, img):
        w, h = img.size
        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2, (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    # 我们在遍历数据集中返回的每一条数据
    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]     # 读取每一条数据，得到图片路径和标签值
        img = Image.open(img_path)      # 利用 Pillow打开图片
        img = img.convert('RGB')        # 将图片转变为RGB格式
        img = self.padding_black(img)
        if self.train_flag:             # 对训练集和测试集分别处理
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)

        return img, label        # 返回图片和其标签值

    # 我们在遍历数据集时，遍历多少，返回的是数据集的长度
    def __len__(self):
        return len(self.imgs_info)

if __name__ == "__main__":
    train_dataset = Data_Loader("mobilenet_train.txt", True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)

    test_dataset = Data_Loader("mobilenet_test.txt", False)
    print("数据个数：", len(test_dataset))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=50, shuffle=True)

    for image, label in test_loader:
        print(image.shape)
        print(label)

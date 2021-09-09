#!/usr/bin/python
# -*- coding:utf-8 -*-
# ------------------------------------------------- #
#      作者：赵泽荣
#      时间：2021年9月9日（农历八月初三）
#      个人站点：1.https://zhao302014.github.io/
#              2.https://blog.csdn.net/IT_charge/
#      个人GitHub地址：https://github.com/zhao302014
# ------------------------------------------------- #
import torch
from torch import nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------- #
#  自己搭建一个 Vgg16 模型结构
#   · 提出时间：2014 年
#   · VGGNet 的网络结构被分为 11，13，16，19 层，该实例实现 16 层的 VGGNet
#   · VGGNet 网络深，卷积层多
#   · 卷积核都是 3* 3 的或 1* 1 的，且同一层中 channel 的数量都是相同的。最大池化层全是 2*2。
#   · 每经过一个 pooling 层，channel 的数量会乘上2（即每次池化之后，Feature Map宽高降低一半，通道数量增加一倍）
#   · 意义：1.证明了更深的网络，能更好的提取特征；2.成为了后续很多网络的 backbone。
#   · 基准 Vgg16 截止到下述代码的 f16 层；由于本实例是手写数字识别（10分类问题），故再后续了一层全连接层 f_output
# --------------------------------------------------------------------------------- #
class MyVgg16Net(nn.Module):
    def __init__(self):
        super(MyVgg16Net, self).__init__()
        self.ReLU = nn.ReLU()   # 论文中的表格，每一大行对应是一个隐藏层,每个隐藏层计算完后的结果都需要经过 ReLU 激活函数进行激活
        # 第一段卷积神经网络：共 3 层，由 2 个卷积层和 1 个最大池化层构成
        self.c1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)    # (224 - 3 + 2*1) / 1 + 1 = 224
        self.c2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)   # (224 - 3 + 2*2) / 1 + 1 = 224
        self.s1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二段卷积神经网络：共 3 层，由 2 个卷积层和 1 个最大池化层构成
        self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)   # (112 - 3 + 2*1) / 1 + 1 = 112
        self.c4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # (112 - 3 + 2*1) / 1 + 1 = 112
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第三段卷积神经网络：共 4 层，由 3 个卷积层和 1 个最大池化层构成
        self.c5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # (56 - 3 + 2*1) / 1 + 1 = 56
        self.c6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # (56 - 3 + 2*1) / 1 + 1 = 56
        self.c7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # (56 - 3 + 2*1) / 1 + 1 = 56
        self.s3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第四段卷积神经网络：共 4 层，由 3 个卷积层和 1 个最大池化层构成
        self.c8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)  # (28 - 3 + 2*1) / 1 + 1 = 28
        self.c9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # (28 - 3 + 2*1) / 1 + 1 = 28
        self.c10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # (28 - 3 + 2*1) / 1 + 1 = 28
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第五段卷积神经网络：共 4 层，由 3 个卷积层和 1 个最大池化层构成
        self.c11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # (28 - 3 + 2*1) / 1 + 1 = 28
        self.c12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # (28 - 3 + 2*1) / 1 + 1 = 28
        self.c13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # (28 - 3 + 2*1) / 1 + 1 = 28
        self.s5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 3 个全连接层置于 5 段卷积层之后
        self.flatten = nn.Flatten()
        self.f14 = nn.Linear(7*7*512, 4096)
        self.f15 = nn.Linear(4096, 4096)
        self.f16 = nn.Linear(4096, 1000)
        # 为满足该实例另加 ↓
        self.f_output = nn.Linear(1000, 10)

    def forward(self, x):                   # 输入shape: torch.Size([1, 3, 224, 224])
        x = self.c1(x)                      # shape: torch.Size([1, 64, 224, 224])
        x = self.c2(x)                      # shape: torch.Size([1, 64, 224, 224])
        x = self.s1(x)                      # shape: torch.Size([1, 64, 112, 112])
        x = self.ReLU(x)
        x = self.c3(x)                      # shape: torch.Size([1, 128, 112, 112])
        x = self.c4(x)                      # shape: torch.Size([1, 128, 112, 112])
        x = self.s2(x)                      # shape: torch.Size([1, 128, 56, 56])
        x = self.ReLU(x)
        x = self.c5(x)                      # shape: torch.Size([1, 256, 56, 56])
        x = self.c6(x)                      # shape: torch.Size([1, 256, 56, 56])
        x = self.c7(x)                      # shape: torch.Size([1, 256, 56, 56])
        x = self.s3(x)                      # shape: torch.Size([1, 256, 28, 28])
        x = self.ReLU(x)
        x = self.c8(x)                      # shape: torch.Size([1, 512, 28, 28])
        x = self.c9(x)                      # shape: torch.Size([1, 512, 28, 28])
        x = self.c10(x)                     # shape: torch.Size([1, 512, 28, 28])
        x = self.s4(x)                      # shape: torch.Size([1, 512, 14, 14])
        x = self.ReLU(x)
        x = self.c11(x)                     # shape: torch.Size([1, 512, 14, 14])
        x = self.c12(x)                     # shape: torch.Size([1, 512, 14, 14])
        x = self.c13(x)                     # shape: torch.Size([1, 512, 14, 14])
        x = self.s5(x)                      # shape: torch.Size([1, 512, 7, 7])
        x = self.ReLU(x)
        x = self.flatten(x)                 # shape: torch.Size([1, 25088])
        x = self.f14(x)                     # shape: torch.Size([1, 4096])
        x = self.f15(x)                     # shape: torch.Size([1, 4096])
        x = self.f16(x)                     # shape: torch.Size([1, 1000])
        # 为满足该实例另加 ↓
        x = self.f_output(x)                # shape: torch.Size([1, 10])
        # 全连接层之后使用了 softmax
        x = F.softmax(x, dim=1)             # shape: torch.Size([1, 10])
        return x


if __name__ == '__main__':
    x = torch.rand([1, 3, 224, 224])
    model = MyVgg16Net()
    y = model(x)

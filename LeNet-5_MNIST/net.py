#!/usr/bin/python
# -*- coding:utf-8 -*-
# ------------------------------------------------- #
#      作者：赵泽荣
#      时间：2021年9月9日（农历八月初三）
#      个人站点：1.https://zhao302014.github.io/
#               2.https://blog.csdn.net/IT_charge/
#      个人GitHub地址：https://github.com/zhao302014
# ------------------------------------------------- #
import torch
from torch import nn

# ------------------------------------------------------------------------------- #
#  自己搭建一个 LeNet-5 模型结构
#   · LeNet-5 是 Yann LeCun 在 1998 年设计的用于手写数字识别的卷积神经网络
#   · 所有卷积核均为 5×5，步长为 1
#   · 所有池化方法为平均池化
#   · 所有激活函数采用 Sigmoid
#   · 该模型共 7 层（3 个卷积层，2 个池化层，2 个全连接层）
#   · LeNet5 网络结构被称为第 1 个典型的 CNN
# ------------------------------------------------------------------------------- #
class MyLeNet5(nn.Module):
    def __init__(self):
        super(MyLeNet5, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.Sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):                # 输入shape: torch.Size([1, 1, 28, 28])
        x = self.Sigmoid(self.c1(x))     # shape: torch.Size([1, 6, 28, 28])
        x = self.s2(x)                   # shape: torch.Size([1, 6, 14, 14])
        x = self.Sigmoid(self.c3(x))     # shape: torch.Size([1, 16, 10, 10])
        x = self.s4(x)                   # shape: torch.Size([1, 16, 5, 5]
        x = self.c5(x)                   # shape: torch.Size([1, 120, 1, 1])
        x = self.flatten(x)              # shape: torch.Size([1, 120])
        x = self.f6(x)                   # shape: torch.Size([1, 84])
        x = self.output(x)               # shape: torch.Size([1, 10])
        return x


if __name__ == '__main__':
    x = torch.rand([1, 1, 28, 28])
    model = MyLeNet5()
    y = model(x)

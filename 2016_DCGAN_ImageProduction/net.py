#!/usr/bin/python
# -*- coding:utf-8 -*-
# ------------------------------------------------- #
#      作者：赵泽荣
#      时间：2021年9月15日（农历八月初九）
#      个人站点：1.https://zhao302014.github.io/
#              2.https://blog.csdn.net/IT_charge/
#      个人GitHub地址：https://github.com/zhao302014
# ------------------------------------------------- #
import torch.nn as nn

# --------------------------------------------------------------------------------- #
#  自己搭建一个 DCGAN 模型结构
#   · DCGAN 提出时间：2016 年
#   · DCGAN 的判别器和生成器都使用了卷积神经网络（CNN）来替代 GAN 中的多层感知机
#   · DCGAN 为了使整个网络可微，拿掉了 CNN 中的池化层，另外将全连接层以全局池化层替代以减轻计算量
#   · DCGAN 相比于 GAN 或者是普通 CNN 的改进包含以下几个方面：1.使用卷积和去卷积代替池化层
#                                                   2.在生成器和判别器中都添加了批量归一化操作
#                                                   3.去掉了全连接层，使用全局池化层替代
#                                                   4.生成器的输出层使用 Tanh 激活函数，其他层使用 RELU
#                                                   5.判别器的所有层都是用 LeakyReLU 激活函数
# --------------------------------------------------------------------------------- #
# 定义生成器结构模型
class MyGenerator(nn.Module):
    def __init__(self):
        super(MyGenerator, self).__init__()
        # 5 个转置卷积层，4 个 BN 层，4 个 ReLU 层，1 个 Tanh 输出激活层
        self.ReLU = nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.Tanh = nn.Tanh()

    def forward(self, x):          # 输入shape: torch.Size([1, 100, 1, 1])
        x = self.deconv1(x)        # shape: torch.Size([1, 1024, 4, 4])
        x = self.bn1(x)            # shape: torch.Size([1, 1024, 4, 4])
        x = self.ReLU(x)           # shape: torch.Size([1, 1024, 4, 4])
        x = self.deconv2(x)        # shape: torch.Size([1, 512, 8, 8])
        x = self.bn2(x)            # shape: torch.Size([1, 512, 8, 8])
        x = self.ReLU(x)           # shape: torch.Size([1, 512, 8, 8])
        x = self.deconv3(x)        # shape: torch.Size([1, 256, 16, 16])
        x = self.bn3(x)            # shape: torch.Size([1, 256, 16, 16])
        x = self.ReLU(x)           # shape: torch.Size([1, 256, 16, 16])
        x = self.deconv4(x)        # shape: torch.Size([1, 128, 32, 32])
        x = self.bn4(x)            # shape: torch.Size([1, 128, 32, 32])
        x = self.ReLU(x)           # shape: torch.Size([1, 128, 32, 32])
        x = self.deconv5(x)        # shape: torch.Size([1, 3, 64, 64])
        x = self.Tanh(x)           # shape: torch.Size([1, 3, 64, 64])
        return x

# 定义判别器结构模型
class MyDiscriminator(nn.Module):
    def __init__(self):
        super(MyDiscriminator, self).__init__()
        # 4 个卷积层，3 个 BN 层，4 个 LRelu 层，1 个 Sigmoid 输出激活层
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.Sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1)

    def forward(self, x):            # 输入shape: torch.Size([1, 3, 64, 64])
        x = self.conv1(x)            # shape: torch.Size([1, 64, 32, 32])
        x = self.LeakyReLU(x)        # shape: torch.Size([1, 64, 32, 32])
        x = self.conv2(x)            # shape: torch.Size([1, 128, 16, 16])
        x = self.bn2(x)              # shape: torch.Size([1, 128, 16, 16])
        x = self.LeakyReLU(x)        # shape: torch.Size([1, 128, 16, 16])
        x = self.conv3(x)            # shape: torch.Size([1, 256, 8, 8])
        x = self.bn3(x)              # shape: torch.Size([1, 256, 8, 8])
        x = self.LeakyReLU(x)        # shape: torch.Size([1, 256, 8, 8])
        x = self.conv4(x)            # shape: torch.Size([1, 512, 4, 4])
        x = self.bn4(x)              # shape: torch.Size([1, 512, 4, 4])
        x = self.LeakyReLU(x)        # shape: torch.Size([1, 512, 4, 4])
        x = self.conv5(x)            # shape: torch.Size([1, 1, 1, 1])
        x = self.Sigmoid(x)          # shape: torch.Size([1, 1, 1, 1])
        return x

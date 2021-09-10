#!/usr/bin/python
# -*- coding:utf-8 -*-
# ------------------------------------------------- #
#      作者：赵泽荣
#      时间：2021年9月10日（农历八月初四）
#      个人站点：1.https://zhao302014.github.io/
#              2.https://blog.csdn.net/IT_charge/
#      个人GitHub地址：https://github.com/zhao302014
# ------------------------------------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------- #
#  自己搭建一个 ResNet18 模型结构
#   · 提出时间：2015 年（作者：何凯明）
#   · ResNet 解决了深度 CNN 模型难训练的问题
#   · ResNet 在 2015 名声大噪,而且影响了 2016 年 DL 在学术界和工业界的发展方向
#   · ResNet 网络是参考了 VGG19 网络，在其基础上进行了修改，并通过短路机制加入了残差单元
#   · 变化主要体现在 ResNet 直接使用 stride=2 的卷积做下采样，并且用 global average pool 层替换了全连接层
#   · ResNet 的一个重要设计原则是：当 feature map 大小降低一半时，feature map 的数量增加一倍，这保持了网络层的复杂度
#   · ResNet18 的 18 指定的是带有权重的 18 层，包括卷积层和全连接层，不包括池化层和 BN 层
#   · ResNet “跳层链接” 的代码体现在相同大小和相同特征图之间用 “+” 相连，而不是 concat
# --------------------------------------------------------------------------------- #
class MyResNet18(nn.Module):
    def __init__(self):
        super(MyResNet18, self).__init__()
        # 第一层：卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        # Max Pooling 层
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 第二、三层：“实线”卷积层
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # 第四、五层：“实线”卷积层
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        # 第六、七层：“虚线”卷积层
        self.conv6_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn6_1 = nn.BatchNorm2d(128)
        self.conv7_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.bn7 = nn.BatchNorm2d(128)
        # 第八、九层：“实线”卷积层
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(128)
        # 第十、十一层：“虚线”卷积层
        self.conv10_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn10_1 = nn.BatchNorm2d(256)
        self.conv11_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn11_1 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0)
        self.bn11 = nn.BatchNorm2d(256)
        # 第十二 、十三层：“实线”卷积层
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn13 = nn.BatchNorm2d(256)
        # 第十四、十五层：“虚线”卷积层
        self.conv14_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn14_1 = nn.BatchNorm2d(512)
        self.conv15_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn15_1 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0)
        self.bn15 = nn.BatchNorm2d(512)
        # 第十六 、十七层：“实线”卷积层
        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(512)
        self.conv17 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn17 = nn.BatchNorm2d(512)
        # avg pooling 层
        self.s2 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        # 第十八层：全连接层
        self.Flatten = nn.Flatten()
        self.f18 = nn.Linear(512, 1000)
        # 为满足该实例另加 ↓
        self.f_output = nn.Linear(1000, 10)

    def forward(self, x):              # shape: torch.Size([1, 3, 224, 224])
        x = self.conv1(x)              # shape: torch.Size([1, 64, 112, 112])
        x = self.bn1(x)                # shape: torch.Size([1, 64, 112, 112])
        x = self.s1(x)                 # shape: torch.Size([1, 64, 56, 56])
        x = self.conv2(x)              # shape: torch.Size([1, 64, 56, 56])
        x = self.bn2(x)                # shape: torch.Size([1, 64, 56, 56])
        x = self.conv3(x)              # shape: torch.Size([1, 64, 56, 56])
        x = self.bn3(x)                # shape: torch.Size([1, 64, 56, 56])
        x = self.conv4(x)              # shape: torch.Size([1, 64, 56, 56])
        x = self.bn4(x)                # shape: torch.Size([1, 64, 56, 56])
        x = self.conv5(x)              # shape: torch.Size([1, 64, 56, 56])
        x = self.bn5(x)                # shape: torch.Size([1, 64, 56, 56])
        x6_1 = self.conv6_1(x)         # shape: torch.Size([1, 128, 28, 28])
        x7_1 = self.conv7_1(x6_1)      # shape: torch.Size([1, 128, 28, 28])
        x7 = self.conv7(x)             # shape: torch.Size([1, 128, 28, 28])
        x = x7 + x7_1                  # shape: torch.Size([1, 128, 28, 28])
        x = self.conv8(x)              # shape: torch.Size([1, 128, 28, 28])
        x = self.conv9(x)              # shape: torch.Size([1, 128, 28, 28])
        x10_1 = self.conv10_1(x)       # shape: torch.Size([1, 256, 14, 14])
        x11_1 = self.conv11_1(x10_1)   # shape: torch.Size([1, 256, 14, 14])
        x11 = self.conv11(x)           # shape: torch.Size([1, 256, 14, 14])
        x = x11 + x11_1                # shape: torch.Size([1, 256, 14, 14])
        x = self.conv12(x)             # shape: torch.Size([1, 256, 14, 14])
        x = self.conv13(x)             # shape: torch.Size([1, 256, 14, 14])
        x14_1 = self.conv14_1(x)       # shape: torch.Size([1, 512, 7, 7])
        x15_1 = self.conv15_1(x14_1)   # shape: torch.Size([1, 512, 7, 7])
        x15 = self.conv15(x)           # shape: torch.Size([1, 512, 7, 7])
        x = x15 + x15_1                # shape: torch.Size([1, 512, 7, 7])
        x = self.conv16(x)             # shape: torch.Size([1, 512, 7, 7])
        x = self.conv17(x)             # shape: torch.Size([1, 512, 7, 7])
        x = self.s2(x)                 # shape: torch.Size([1, 512, 1, 1])
        x = self.Flatten(x)            # shape: shape: torch.Size([1, 512])
        x = self.f18(x)                # shape: torch.Size([1, 1000])
        # 为满足该实例另加 ↓
        x = self.f_output(x)           # shape: torch.Size([1, 10])
        x = F.softmax(x, dim=1)        # shape: torch.Size([1, 10])
        return x

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = MyResNet18()
    y = model(x)

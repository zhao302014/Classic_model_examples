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
from net import MyAlexNet
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

data_transform = transforms.Compose([
    transforms.Scale(224),     # 缩放图像大小为 224*224
    transforms.ToTensor()      # 仅对数据做转换为 tensor 格式操作
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
# 给训练集创建一个数据集加载器
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
# 给测试集创建一个数据集加载器
test_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用 net 里定义的模型，如果 GPU 可用则将模型转到 GPU
model = MyAlexNet().to(device)
# 加载 train.py 里训练好的模型
model.load_state_dict(torch.load("./save_model/99model.pth"))

# 获取预测结果
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

# 把 tensor 转成 Image，方便可视化
show = ToPILImage()
# 进入验证阶段
model.eval()
# 对 test_dataset 里 10000 张手写数字图片进行推理
for i in range(len(test_dataset)):
    x, y = test_dataset[i][0], test_dataset[i][1]
    # tensor格式数据可视化
    show(x).show()
    # 扩展张量维度为 4 维
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)
    # 单通道转为三通道
    x = x.cpu()
    x = np.array(x)
    x = x.transpose((1, 0, 2, 3))          # array 转置
    x = np.concatenate((x, x, x), axis=0)
    x = x.transpose((1, 0, 2, 3))      # array 转置回来
    x = torch.tensor(x).to(device)   # 将 numpy 数据格式转为 tensor，并转回 cuda 格式
    with torch.no_grad():
        pred = model(x)
        # 得到预测类别中最高的那一类，再把最高的这一类对应classes中的哪一个标签
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        # 最终输出预测值与真实值
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

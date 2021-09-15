#!/usr/bin/python
# -*- coding:utf-8 -*-
# ------------------------------------------------- #
#      作者：赵泽荣
#      时间：2021年9月15日（农历八月初九）
#      个人站点：1.https://zhao302014.github.io/
#              2.https://blog.csdn.net/IT_charge/
#      个人GitHub地址：https://github.com/zhao302014
# ------------------------------------------------- #
import numpy as np
import torch.utils.data
import torchvision.utils as vutils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.datasets as dset
from net import MyGenerator

data_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 加载训练数据集
train_data_path = "./data/"
train_dataset = dset.ImageFolder(root=train_data_path, transform=data_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用 net 里定义的模型，如果 GPU 可用则将模型转到 GPU
modelGenerator = MyGenerator().to(device)
# 加载 train.py 里训练好的模型
modelGenerator.load_state_dict(torch.load("./save_model/49model.pt"))    # 注：一般而言，50轮之后的模型才能生成较好“假”图

# 进入验证阶段
modelGenerator.eval()
img_list = []
iters = 0

# 开始验证
for i, data in enumerate(train_dataloader):
    noise = torch.randn(64, 100, 1, 1, device=device)
    if (iters % 20 == 0):
        fake_img = modelGenerator(noise).cpu()
        img_list.append(vutils.make_grid(fake_img, normalize=True))
    iters += 1

# 真图与生成图对比
real_batch = next(iter(train_dataloader))
real_img = vutils.make_grid(real_batch[0], normalize=True).cpu()
# 开始画图
plt.figure(figsize=(12, 12))    # 设置画布“宽、长”大小（单位为inch）
# 一行两列，真图位于左边位置
plt.subplot(1, 2, 1)
plt.axis("off")       # 关闭坐标轴
plt.title("Real Images")      # 设置图像标题
plt.imshow(np.transpose(real_img, (1, 2, 0)))
# 一行两列，生成图位于右边位置
plt.subplot(1, 2, 2)
plt.axis("off")       # 关闭坐标轴
plt.title("Fake Images")      # 设置图像标题
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()      # 图像显示

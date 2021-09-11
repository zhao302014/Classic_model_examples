#!/usr/bin/python
# -*- coding:utf-8 -*-
# ------------------------------------------------- #
#      作者：赵泽荣
#      时间：2021年9月11日（农历八月初五）
#      个人站点：1.https://zhao302014.github.io/
#              2.https://blog.csdn.net/IT_charge/
#      个人GitHub地址：https://github.com/zhao302014
# ------------------------------------------------- #
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from net import MyUNet
from CreateDataset import DataLoader

# 加载训练数据集
train_data_path = "data/train/"
train_dataset = DataLoader(train_data_path)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

# 如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用 net 里定义的模型，如果 GPU 可用则将模型转到 GPU
model = MyUNet().to(device)

# 定义损失函数
loss_fn = nn.BCEWithLogitsLoss()
# 定义优化器
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9)
# 学习率每隔 10 个 epoch 变为原来的 0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        # 前向传播
        x, y = X.to(device), y.to(device)
        x = x.float()  # 输入的类型是字节型的 tensor，而加载的权重的类型是 float 类型的 tensor，需要将字节型的 tensor 转化为 float 型的 tensor
        y = y.float()
        output = model(x)
        cur_loss = loss_fn(output, y)
        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        n = n + 1
    print('train_loss：' + str(loss / n))

# 开始训练
epoch = 100
for t in range(epoch):
    lr_scheduler.step()
    print(f"Epoch {t + 1}\n----------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    torch.save(model.state_dict(), "save_model/{}model.pth".format(t))    # 模型保存
print("Done!")

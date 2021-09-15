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
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler
import torchvision.datasets as dset
import torchvision.transforms as transforms
from net import MyGenerator, MyDiscriminator

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
modelDiscriminator = MyDiscriminator().to(device)

# 定义损失函数
loss = nn.BCELoss()
# 定义优化器
optimizerGenerator = optim.Adam(modelGenerator.parameters(), lr=0.0003, betas=(0.5, 0.999))
optimizerDiscriminator = optim.Adam(modelDiscriminator.parameters(), lr=0.0003, betas=(0.5, 0.999))
# 学习率每隔 10 个 epoch 变为原来的 0.1
lr_scheduler_Generator = lr_scheduler.StepLR(optimizerGenerator, step_size=10, gamma=0.1)
lr_scheduler_Discriminator = lr_scheduler.StepLR(optimizerDiscriminator, step_size=10, gamma=0.1)

# 定义训练函数
discriminator_loss, generator_loss = 0, 0
def train(train_dataloader, modelGenerator, modelDiscriminator, loss, optimizerGenerator, optimizerDiscriminator):
    for i, data in enumerate(train_dataloader, 0):
        # ----------------------------------------- #
        #   更新判别器：最大化 log(D(x)) + log(1 - D(G(z)))
        # ----------------------------------------- #
        # 用所有“真”的数据进行训练
        modelDiscriminator.zero_grad()     # 梯度清空
        real_img = data[0].to(device)      # data[0].shape: torch.Size([batch_size, c, w, h])
        batch_size = real_img.size(0)      # 获取 batch_size 大小
        real_img_label = torch.full((batch_size,), 1.0, device=device)   # 1.0：“真”标签；label：tensor([1., 1., 1., 1., ... , 1., 1., 1., 1.])
        # 判别器推理
        real_img_output = modelDiscriminator(real_img).view(-1)   # shape: torch.Size([64])
        # 计算所有“真”标签的损失函数
        real_img_loss = loss(real_img_output, real_img_label)
        real_img_loss.backward()               # 误差反传
        # 生成假数据并进行训练
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        # 用生成器生成假图像
        fake_img = modelGenerator(noise)
        fake_img_label = torch.full((batch_size,), 0.0, device=device)   # 0.0：“假”标签；label：tensor([0., 0., 0., 0., ... , 0., 0., 0., 0.])
        fake_img_output = modelDiscriminator(fake_img.detach()).view(-1)
        # 计算判别器在假数据上的损失
        fake_img_loss = loss(fake_img_output, fake_img_label)
        fake_img_loss.backward()              # 误差反传
        discriminator_loss = real_img_loss + fake_img_loss
        optimizerDiscriminator.step()     # 参数更新
        # ----------------------------------------- #
        #   更新生成器：最大化 log(D(G(z)))
        # ----------------------------------------- #
        modelGenerator.zero_grad()
        # 生成器样本标签都为 1
        img_label = torch.full((batch_size,), 1.0, device=device)       # 1.0：“真”标签；label：tensor([1., 1., 1., 1., ... , 1., 1., 1., 1.])
        img_output = modelDiscriminator(fake_img).view(-1)
        # 计算损失
        img_loss = loss(img_output, img_label)
        img_loss.backward()            # 误差反传
        optimizerGenerator.step()     # 参数更新
        generator_loss = real_img_loss
    print('判别器loss：', discriminator_loss.item())
    print('生成器loss：', generator_loss.item())

# 开始训练
epoch = 50
for t in range(epoch):
    lr_scheduler_Generator.step()
    lr_scheduler_Discriminator.step()
    print(f"Epoch {t + 1}\n--------------------------------")
    train(train_dataloader, modelGenerator, modelDiscriminator, loss, optimizerGenerator, optimizerDiscriminator)
    torch.save(modelGenerator.state_dict(), "save_model/{}model.pt".format(t))    # 模型保存
print("Done!")

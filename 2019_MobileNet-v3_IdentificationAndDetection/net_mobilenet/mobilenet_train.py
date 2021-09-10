#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.optim as optim
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torch.utils.data import DataLoader
from CreateDataloader import Data_Loader
from tensorboardX import SummaryWriter

# 用编号指定用那个显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 检查是否显卡可用
device = "cuda" if torch.cuda.is_available() else "cpu"

# 保存数据
def save_checkpoint(state, filename):
    torch.save(state, filename)

# 训练代码
def train(train_loader, model, criterion, optimizer, epoch, writer):
    model.train()
    loss = 0.0
    acc = 0.0
    n = 0
    # 读取训练集当中的所有数据, i是遍历的编号，用来控制输出的次数
    for i, (input, target) in enumerate(train_loader):
        # 把输入和标签都传送到device当中
        input = input.to(device)
        target = target.to(device)

        # 把输入传入到model里面，得到结果叫做output
        output = model(input)

        # 比较output和标签之间的差异，用到的criterion是传入的函数，是比较方法
        cur_loss = criterion(output, target)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(target == pred) / output.shape[0]

        # 优化器的“归零”操作
        optimizer.zero_grad()
        # 误差进行反向传播
        cur_loss.backward()
        # 优化器的参数更新
        optimizer.step()
        n = n + 1
        loss += cur_loss.item()
        acc += cur_acc.item()
    print('my_net_train_loss：' + str(loss / n))
    print('my_net_train_acc：' + str(acc / n))
    # 在tensorboard中显示出loss值的大小
    writer.add_scalar('mobilenet_train_loss', loss / n, epoch)
    writer.add_scalar('mobilenet_train_acc', acc / n, epoch)

# 验证函数
def validate(val_loader, model, criterion, epoch, writer, phase="VAL"):
    # 将模型转化到验证模式
    model.eval()
    loss = 0.0
    acc = 0.0
    n = 0
    # 模型的参数都不会进行更新（把模型的参数固定下来）
    with torch.no_grad():
        # 遍历测试数据集
        for i, (input, target) in enumerate(val_loader):
            # 把数据传入到设备中（CPU or GPU）
            input = input.to(device)
            target = target.to(device)

            # 把input传入到模型当中，得到output
            output = model(input)

            # 对output和标签值进行比较
            cur_loss = criterion(output, target)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(target == pred) / output.shape[0]

            n = n + 1
            loss += cur_loss.item()
            acc += cur_acc.item()
        print('my_net_test_loss：' + str(loss / n))
        print('my_net_test_acc：' + str(acc / n))
        # 在tensorboard中显示出loss值的大小
        writer.add_scalar('mobilenet_test_loss', loss / n, epoch)
        writer.add_scalar('mobilenet_test_acc', acc / n, epoch)


if __name__ == "__main__":
    # 1:加载数据
    train_dir_list = 'mobilenet_train.txt'
    valid_dir_list = 'mobilenet_test.txt'

    train_data = Data_Loader(train_dir_list, train_flag=True)      # 自创一个data_loader
    valid_data = Data_Loader(valid_dir_list, train_flag=False)     # test_data是验证集

    train_loader = DataLoader(dataset=train_data, num_workers=0, pin_memory=True, batch_size=16, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, num_workers=0, pin_memory=True, batch_size=16)
    # 2:定义网络
    model = mobilenet_v3_small(pretrained=False)
    print(model)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 5)
    print(model)

    pretrained_dict = torch.load('./mobilenet_v3_pretrain.pth')

    pretrained_dict.pop('classifier.3.weight')
    pretrained_dict.pop('classifier.3.bias')

    # 自己的模型参数变量，在开始时里面参数处于初始状态，所以很多0和1
    model_dict = model.state_dict()
    # 去除一些不需要的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 模型参数列表进行参数更新，加载参数
    model_dict.update(pretrained_dict)

    # 改进过的预训练模型结构，加载刚刚的模型参数列表
    model.load_state_dict(model_dict)

    for name, value in model.named_parameters():
        if (name != 'classifier.3.weight') and (name != 'classifier.3.bias'):
            value.requires_grad = False

    # filter 函数将模型中属性 requires_grad = True 的参数选出来
    params_conv = filter(lambda p: p.requires_grad, model.parameters())    # 要更新的参数在parms_conv当中

    model = model.to(device)
    # 3:定义损失函数和优化器等
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 利用Adam优化算法
    optimizer = optim.Adam(params_conv, lr=0.001, weight_decay=0.001)
    # 学习率每step_size次， 下降0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 我们的tensorboard保存路径的地方
    writer = SummaryWriter('net_mobilenet_logs')
    # 4:开始训练
    epochs = 100  # 总共训练回合数
    for epoch in range(epochs):
        print("第", epoch, "轮")
        # 更新优化器
        scheduler.step()
        # 模型训练
        train(train_loader, model, criterion, optimizer, epoch, writer)
        # 在验证集上测试效果
        validate(valid_loader, model, criterion, epoch, writer, phase="VAL")

    torch.save(model, 'net_mobilenet_v3_model1.pth')
    writer.close()

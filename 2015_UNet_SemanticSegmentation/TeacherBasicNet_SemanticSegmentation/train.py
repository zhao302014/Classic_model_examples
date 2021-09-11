import os

import numpy as np
import torch
from tensorboardX import writer
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from data_procress import SegDataset
from net import simpleNet5

batchsize = 64
epochs = 100
imagesize = 256
cropsize = 224
train_data_path = 'E:\python-project\mouth_check\data'
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
dataset = SegDataset(train_data_path,imagesize,cropsize,data_transform)
dataloader = DataLoader(dataset,batchsize,shuffle=True)

device = torch.device('cuda')
net = simpleNet5().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=1e-2,momentum=0.9)
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoint')

for epochs in range(1,epochs+1):
    for batch_idx,(img,lable) in enumerate(dataloader):
        img,lable = img.to(device).float(),lable.to(device).float()
        output = net(img)
        # print(output.shape)
        # print(lable.shape)

        loss = criterion(output, lable.long())

        output_mask = output.cpu().data.numpy().copy()

        output_mask = np.argmax(output_mask,axis=1)

        lable_mask = lable.cpu().data.numpy().copy()


        acc = (output_mask == lable_mask)

        acc = np.mean(acc)

        # writer.add_scalar('trainloss',loss.item(),epochs)
        # writer.add_scalar('accloss', loss.item(), epochs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch=',epochs)
    print("acc=" ,acc)
    print("loss=",loss)

    if epochs%10 == 0:
        torch.save(net,'checkpoint/model_epoch_{}.pth'.format(epochs))

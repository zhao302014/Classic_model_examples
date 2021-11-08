#!/usr/bin/python
# -*- coding:utf-8 -*-
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import os

# 如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = torch.load("net_mobilenet_v3_model.pth")

val_tf = transforms.Compose([    # 简单把图片压缩了变成Tensor模式
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])    # 图片标准化
])

def padding_black(img):
    w, h = img.size
    scale = 224. / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
    size_fg = img_fg.size
    size_bg = 224
    img_bg = Image.new("RGB", (size_bg, size_bg))
    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                          (size_bg - size_fg[1]) // 2))
    img = img_bg
    return img

dir_loc = "F:/data/mouse"
model.eval()
with torch.no_grad():
    for a, b, c in os.walk(dir_loc):
        for filei in c:
            full_path = os.path.join(a, filei)
            img = Image.open(full_path)   # 打开图片
            img = img.convert('RGB')      # 转换为RGB 格式
            img = padding_black(img)
            img_tensor = val_tf(img)

            # 增加batch_size维度
            img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False).to(device)
            output_tensor = model(img_tensor)

            # 将输出通过softmax变为概率值
            output = torch.softmax(output_tensor, dim=1)

            # 输出可能性最大的那位
            pred_value, pred_index = torch.max(output, 1)

            # 将数据从cuda转回cpu
            if torch.cuda.is_available() == False:
                pred_value = pred_value.detach().cpu().numpy()
                pred_index = pred_index.detach().cpu().numpy()

            # 增加类别标签
            classes = ["蚂蚁", "蟑螂", "苍蝇", "老鼠", "鼠妇"]

            print("预测类别为： ", classes[pred_index[0]], " 可能性为: ", pred_value[0] * 100, "%")

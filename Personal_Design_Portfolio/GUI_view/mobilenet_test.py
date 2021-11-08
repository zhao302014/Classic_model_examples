#!/usr/bin/python
# -*- coding:utf-8 -*-
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

def detect(path):
    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load("../net_mobilenet/net_mobilenet_v3_model.pth")

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

    img_path = path
    model.eval()
    with torch.no_grad():
        img = Image.open(img_path)   # 打开图片
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
    return classes[pred_index[0]], pred_value[0].item() * 100

if __name__ == '__main__':
    path = ''
    pred_class, pred_value = detect(path)
    print('预测类别为：', pred_class)
    print('预测概率为：', pred_value)
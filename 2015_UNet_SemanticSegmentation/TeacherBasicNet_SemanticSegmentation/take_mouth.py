import dlib
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
modelpath = sys.argv[1]
net = torch.load(modelpath)
net.eval()
torch.no_grad()

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# 配置 Dlib 关键点检测路径
# 文件可以从 http://dlib.net/files/ 下载
PREDICTOR_PATH = "E:\Anaconda\envs\pytorch\Lib\site-packages\cv2\data\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
# 配置人脸检测器路径
cascade_path = 'E:\Anaconda\envs\pytorch\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

# 调用 cascade.detectMultiScale 人脸检测器和 Dlib 的关键点检测算法 predictor 获得关键点结果
def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.3, 5) # 人脸检测
    x, y, w, h = rects[0]  # 获取人脸的四个属性值，左上角坐标 x,y 、高宽 w、h
#     print(x, y, w, h)
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im,
                    str(idx),
                    pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 5, color=(0, 255, 255))
    return im

def getlipfromimage(im, landmarks):
    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0
    # 根据最外围的关键点获取包围嘴唇的最小矩形框
    # 68 个关键点是从
    # 左耳朵0 -下巴-右耳朵16-左眉毛（17-21）-右眉毛（22-26）-左眼睛（36-41）
    # 右眼睛（42-47）-鼻子从上到下（27-30）-鼻孔（31-35）
    # 嘴巴外轮廓（48-59）嘴巴内轮廓（60-67）
    for i in range(48, 67):
        x = landmarks[i, 0]
        y = landmarks[i, 1]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y


    # print("xmin=", xmin)
    # print("xmax=", xmax)
    # print("ymin=", ymin)
    # print("ymax=", ymax)

    roiwidth = xmax - xmin
    roiheight = ymax - ymin

    roi = im[ymin:ymax, xmin:xmax, 0:3]

    if roiwidth > roiheight:
        dstlen = 1.5 * roiwidth
    else:
        dstlen = 1.5 * roiheight

    diff_xlen = dstlen - roiwidth
    diff_ylen = dstlen - roiheight

    newx = xmin
    newy = ymin

    imagerows, imagecols, channel = im.shape
    if newx >= diff_xlen / 2 and newx + roiwidth + diff_xlen / 2 < imagecols:
        newx = newx - diff_xlen / 2
    elif newx < diff_xlen / 2:
        newx = 0
    else:
        newx = imagecols - dstlen

    if newy >= diff_ylen / 2 and newy + roiheight + diff_ylen / 2 < imagerows:
        newy = newy - diff_ylen / 2
    elif newy < diff_ylen / 2:
        newy = 0
    else:
        newy = imagerows - dstlen

    roi = im[int(newy):int(newy) + int(dstlen), int(newx):int(newx) + int(dstlen), 0:3]
    return roi,int(newy),int(newx),dstlen

def listfiles(rootDir):
    list_dirs = os.walk(rootDir)
    for root, dirs, files in list_dirs:
        for d in dirs:
            print(os.path.join(root, d))
        for f in files:
            fileid = f.split('.')[0]
            filepath = os.path.join(root, f)

            try:
                im = cv2.imread(filepath, 1)
                landmarks = get_landmarks(im)
                roi,offsety,offsetx,dstlen = getlipfromimage(im, landmarks)
                image = (cv2.resize(roi, (224, 224), interpolation=cv2.INTER_NEAREST) / 255.0).astype(np.float32)
                imgblob = data_transforms(image).unsqueeze(0)
                imgblob = Variable(imgblob).cuda()
                predict = F.softmax(net(imgblob)).cpu().data.numpy().copy()
                predict = np.argmax(predict, axis=1)
                result = np.squeeze(predict)
                result = (result * 255).astype(np.uint8)

                image[result>0] = [0,0,255]

                image = (cv2.resize(image, (roi.shape[0], roi.shape[1]), interpolation=cv2.INTER_NEAREST))


                imt = im.copy()
                im[int(offsety):int(offsety + dstlen), int(offsetx):int(offsetx + dstlen), 0:3] = image
                im[im == 0] = imt[im == 0]
                # cv2.imwrite(os.path.join(sys.argv[2], filepath.split('.')[-2]+'_seg.jpg'), result)
                cv2.imwrite(os.path.join(sys.argv[2], f), im)


            except:
                continue

listfiles("E:\python-project\mouth_check\input\image")
import os
import random
import torchvision.transforms.functional as TF
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

class SegDataset(Dataset):
    def __init__(self, traindir, imagesize, cropsize,transform):
        self.imagedir = os.path.join(traindir,'img')
        self.labeldir = os.path.join(traindir,'label')
        self.images = os.listdir(self.imagedir)
        self.labels = os.listdir(self.labeldir)
        self.imagesize = imagesize
        self.cropsize = cropsize
        assert len(self.images) == len(self.labels)
        self.transform = transform
        self.samples = []
        for i in range(len(self.images)):
            self.samples.append((os.path.join(self.imagedir,self.images[i]),os.path.join(self.labeldir,self.labels[i])))

    def __getitem__(self, item):
        img_path, label_path = self.samples[item]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.imagesize, self.imagesize), interpolation=cv2.INTER_NEAREST).astype(np.float) / 255.0
        label = (cv2.imread(label_path, 0) > 0).astype(np.uint8)
        label = cv2.resize(label, (self.imagesize, self.imagesize), interpolation=cv2.INTER_NEAREST)
        randoffsetx = np.random.randint(self.imagesize - self.cropsize)
        randoffsety = np.random.randint(self.imagesize - self.cropsize)
        img = img[randoffsety:randoffsety + self.cropsize, randoffsetx:randoffsetx + self.cropsize]
        label = label[randoffsety:randoffsety + self.cropsize, randoffsetx:randoffsetx + self.cropsize]

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)

    def my_segmentation_transforms(image, segementation):
        if random.random()>0.5:
            angle = random.randint(-30,30)
            image = TF.rotate(image,angle)
            segementation = TF.rotate(segementation,angle)
        return image,segementation


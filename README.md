# 经典网络项目实战
这里存放了一些经典网络及其项目实战内容。</br>
具体而言，是按照 CNN 发展时间节点来拟写存放的。</br>
</br>
截至目前，编写完成的项目有：</br>
 - <b>1998 年——LeNet-5</b>：这是 Yann LeCun 在 1998 年设计的用于手写数字识别的卷积神经网络。该项目自己搭建了 Lenet-5 网络并在 MNIST 手写数字识别项目中得到了应用。（注：该项目最令我印象深刻的是我自己验证了几年前学者验证的最大池化的效果是要优于平均池化的；这一点在本项目代码中并没有体现，原因是项目旨在遵循基准 LeNet-5 模型的各项指标，应用了基准模型设计的平均池化，若改为最大池化，训练代码中的优化器定义可删去动量项，亦可随之删去学习率变化代码，也可达到同样效果）

<div align="center"><img src="https://www.researchgate.net/profile/Sheraz-Khan-14/publication/321586653/figure/fig4/AS:568546847014912@1512563539828/The-LeNet-5-Architecture-a-convolutional-neural-network.png" /></div> 

 - <b>2012 年——AlexNet</b>：这是 2012 年 ImageNet 竞赛冠军获得者 Hinton 和他的学生 Alex Krizhevsky 设计的。该项目自己搭建了 AlexNet 网络并在 MNIST 手写数字识别项目中得到了应用。（注：MNIST 手写数字识别数据集是单通道的，在该项目中用 numpy 库将图片依次转换为 3 通道在进行处理）

<div align="center"><img src="https://miro.medium.com/proxy/1*qyc21qM0oxWEuRaj-XJKcw.png" /></div> 

 - <b>2014 年——VGGNet</b>：这是 2014 年牛津大学计算机视觉组和 Google DeepMind 公司研究员一起研发的深度网络模型。该网络结构被分为 11，13，16，19 层；该项目自己搭建了 VGGNet 网络并在 MNIST 手写数字识别项目中得到了应用。（注：该项目主要修改了 AlexNet 应用实例中的 net.py 代码，由于输入图片通道数依然为 3 通道，所以延续了 AlexNet 应用实例中的 train.py 与 test.py ，仅调小了其中的 banch_size （由 16 变为了 8），以避免因 CUDA 内存不足而引起的报错）

<div align="center"><img src="https://www.researchgate.net/profile/Timea-Bezdan/publication/333242381/figure/fig2/AS:760979981860866@1558443174380/VGGNet-architecture-19.ppm" /></div> 

 - <b>2014 年——GoogLeNet</b>：这是 google 推出的基于 Inception 模块的深度神经网络模型，在 2014 年的 ImageNet 竞赛中夺得了冠军，在随后的两年中一直在改进，形成了 Inception V2、Inception V3、Inception V4 等版本。；该项目自己搭建了 GoogLeNet 网络并在 MNIST 手写数字识别项目中得到了应用。（注：net.py 代码着实很长，原因是冗余的太多了，正像我的朋友吃午饭时讲的那样：“这个代码重复部分很多，完全可以写到一个函数里调用啊，这样写太没有灵魂了。”怎么说呢，主要当时快写完了，后来想想也是；也不能说这样写清晰地展现了 GoogLeNet 的架构，反而会让人觉得这是一个憨批程序员，一根筋，不懂得变通，所以这个自我复现的 GoogLeNet 版本的 net.py 文件代码就这样留着吧，也算是一个提醒，以后写代码在实现功能的基础上争取精简一些...）

<div align="center"><img src="https://miro.medium.com/max/5176/1*ZFPOSAted10TPd3hBQU8iQ.png" /></div> 

 - <b>2019 年——MobileNet-v3</b>：这是 Google 在 2019 年 3 月 21 日提出的网络架构，也是继 MobileNet-v2 之后的又一力作，MobileNet-v3 small 在 ImageNet 分类任务上，较 MobileNet-v2，精度提高了大约 3.2%，时间却减少了 15%，MobileNet-v3 large 在 imagenet 分类任务上，较 MobileNet-v2，精度提高了大约 4.6%，时间减少了 5%，MobileNet-v3 large 与 v2 相比，在 COCO 上达到相同的精度，速度快了 25%，同时在分割算法上也有一定的提高。论文还有一个亮点在于，网络的设计利用了 NAS（network architecture search）算法以及 NetAdapt algorithm 算法。并且，论文还介绍了一些提升网络效果的 trick，这些 trick 也提升了不少的精度以及速度。

<div align="center"><img src="https://1.bp.blogspot.com/-qMBHklyOfic/XcxKvHgiB8I/AAAAAAAAE8A/osT1RxwyqPY7bE_x7vsyYTYiIt7QSn0hQCEwYBhgL/s640/image1.png" /></div> 

</br>

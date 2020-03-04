import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import torchvision

from PIL import Image

import matplotlib.pyplot as plt

train_txt_path = os.path.join("..", "..", "Data")   #地址拼接符号os.path.join

train_bs = 4  # 每次训练图片数
valid_bs = 16
lr_init = 0.001
max_epoch = 5    # 训练遍历数据集次数


# 根据传入地址从硬盘里提取图像数据和标签数据，基本上所有的提取数据的类操作都继承torch库里面的Dataset类：from torch.utils.data import Dataset
class VOCSegmentation(Dataset): 
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    """
    def __init__(self, root ,transform=None, target_transform=None):   # 类的初始化操作函数，包含三个参数。root是传入的地址，transform是图像的变换操作，默认为没有
        super(VOCSegmentation, self).__init__()
        self.root = root 
        self.transform = transform
        voc_root = os.path.join(self.root ,"VOCdevkit", "VOC2012")    # os.path.join地址拼接函数，在传入的地址下面有一个名叫“VOCdevkit”的文件夹，在“VOCdevkit”里面有名叫"VOC2012"的文件夹      
        image_dir = os.path.join(voc_root, 'JPEGImages')      # voc_root目录下的'JPEGImages'，里面是原图像
        mask_dir = os.path.join(voc_root, voc_root目录下的)  # voc_root目录下的voc_root目录下的，里面存放的是分割后的标签

        if not os.path.isdir(voc_root): 
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')   # 里面存放的是图片地址索引

        split_f = os.path.join(splits_dir, 'train.txt')     # 训练集的地址索引

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()] 

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]     # 将txt文件里的原图片名字加上。'.jpg'后缀，self.images是一个列表
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]   # 将txt文件里的标签图片名字加上。'.jpg'后缀，self.images是一个列表
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):    # 这个函数是Dataset包含的固有的，根据index索引返回图片矩阵数据
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')     # 打开原图片数据
        target = Image.open(self.masks[index])              # 打开标签图像数据
        
        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)

        return img, target

    def __len__(self):
        return len(self.images)       # 返回数据集的长度
    
# ------------------------------------ step 1/5 : 加载数据------------------------------------

# 数据预处理设置
normMean = [0.4948052, 0.48568845, 0.44682974]     # 数据预处理，三通道均值
normStd = [0.24580306, 0.24236229, 0.2603115]      # 数据预处理 ，三通道方差 
normTransform = transforms.Normalize(normMean, normStd)   # 标准化处理
trainTransform = transforms.Compose([
    transforms.Resize(32),                              # 重置图像分辨率
    transforms.RandomCrop(32, padding=4),            # 随机剪裁
    transforms.ToTensor(),                       # 将图像数据转换为张量
#    transforms.Lambda(lambda x: x.repeat(3,1,1)),
#    normTransform
])


# 构建MyDataset实例
train_txt_path = os.path.join("..", "..", "Data")   # 这是我们训练数据的存放地址
train_data = VOCSegmentation(root=train_txt_path, transform=trainTransform)     # 实例化数据提取器，提取器是根据txt文件读取图像数据
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True) # 实例化数据加载器，加载器是从提取器里加载到内存，通过对索引的累加达到遍历数据集的作用


# ------------------------------------ step 2/5 : 定义网络------------------------------------

net = torchvision.models.segmentation.deeplabv3_resnet50()  # 这里直接调用的torchvision里的deeplab
#net=net.cuda()


# ------------------------------------ step 3/5 : 定义损失函数和优化器,准确率 ------------------------------------

# 语义分割损失函数，这里采用的是预测与真实值的交集占两者合集的比例

def dice_loss (y_pred ,y_true, smooth=1):
    mean_loss = 0
    for i in range(y_pred.size(-1)):
        intersection = torch.sum( y_true[:,:,:,i] * y_pred[:,:,:,i] )  # 两者交集
        union = torch.sum(y_true[:,:,:,i] ) + torch.sum(y_pred[:,:,:,i] ) # 两者合集
    mean_loss += (2. * intersection +smooth)/(union + smooth)
    return  1-torch.mean(mean_loss, axis=0）

optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)    # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     # 设置学习率下降策略

for epoch in range(1):

    print('Start Training!')
    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0       # 预测正确的像素点个数
    total = 0.0         # 总共的像素点个数

    for i, data in enumerate(train_loader):     # enumerate与Dataloder类配套使用，遍历图像数据集
        inputs, target = data
        inputs, target = Variable(inputs), Variable(target)

        optimizer.zero_grad()    # 清空梯度
        outputs = net(inputs)    # 前向传播
        loss = dice_loss( outputs['out'], target) # 计算损失
        print("loss={}".format(loss))
        loss.backward()         # 反向传播
        optimizer.step()        #  更新参数
        
        predicted = torch.argmax(outputs['out'],dim=1)      # outputs['out']通道数为21维，分别对应21种分类的概率，取出此次网络推测类别概率最高的类，返回索引值
        target = target*255                                 # voc数据标签类别共21类，将其标签值设为0到20的整数
        target = torch.round(target.squeeze())。            # 四舍五入

        correct += (predicted == target).squeeze().sum().numpy()   # 统计正确分类的像素点个数
        total += target.numel()                             # torch.numel()返回张量中元素个数，这里返回像素点个数
        print("correct={}".format(correct))

    accurcy = correct/total                # 平均正确率
    print("epoch={}".format(epoch))
    print("epoch_accucy={}".format(accurcy))

        

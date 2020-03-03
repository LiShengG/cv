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

train_bs = 4
valid_bs = 16
lr_init = 0.001
max_epoch = 5


class VOCSegmentation(Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    """
    def __init__(self, root ,transform=None, target_transform=None):
        super(VOCSegmentation, self).__init__()
        self.root = root 
        self.transform = transform
        voc_root = os.path.join(self.root ,"VOCdevkit", "VOC2012")
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, 'train.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        
        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)

        return img, target

    def __len__(self):
        return len(self.images)
    
    


# ------------------------------------ step 1/5 : 加载数据------------------------------------

# 数据预处理设置
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normTransform = transforms.Normalize(normMean, normStd)
trainTransform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
#    transforms.Lambda(lambda x: x.repeat(3,1,1)),
#    normTransform
])


# 构建MyDataset实例
train_txt_path = os.path.join("..", "..", "Data")
train_data = VOCSegmentation(root=train_txt_path, transform=trainTransform)
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)


# ------------------------------------ step 2/5 : 定义网络------------------------------------

net = torchvision.models.segmentation.deeplabv3_resnet50()
#net=net.cuda()


# ------------------------------------ step 3/5 : 定义损失函数和优化器,准确率 ------------------------------------

# 语义分割损失函数

def dice_loss (y_pred ,y_true, smooth=1):
    mean_loss = 0
    for i in range(y_pred.size(-1)):
        intersection = torch.sum( y_true[:,:,:,i] * y_pred[:,:,:,i] )
        union = torch.sum(y_true[:,:,:,i] ) + torch.sum(y_pred[:,:,:,i] )
    mean_loss += (2. * intersection +smooth)/(union + smooth)
    return  1-torch.mean(mean_loss, axis=0)


optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)    # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
# 设置学习率下降策略

for epoch in range(1):

    print('Start Training!')
    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0

    for i, data in enumerate(train_loader):
        inputs, target = data
        inputs, target = Variable(inputs), Variable(target)

        optimizer.zero_grad()
        outputs = net(inputs)
#        print("outputs.size={}".format(outputs['out'].size()))

        loss = dice_loss( outputs['out'], target)
        print("loss={}".format(loss))
        loss.backward()
        optimizer.step()
        
        predicted = torch.argmax(outputs['out'],dim=1)   
        target = target*255
        target = torch.round(target.squeeze())

        correct += (predicted == target).squeeze().sum().numpy()
        total += target.numel()
        print("correct={}".format(correct))

    accurcy = correct/total
    print("epoch={}".format(epoch))
    print("accucy={}".format(accurcy))

        








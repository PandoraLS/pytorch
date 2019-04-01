# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)  # in_channels=3,out_channels=6,kernel_size=5
        self.pool = nn.MaxPool2d(2, 2)  # kernel_size和stride
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        forward(x) , 指明一个batch的输入数据 x ,流经网络的过程
        :param x: 一个input变量
        :return: x经过Net之后的输出
        """
        # 最初的x：x = [N, 3, 32, 32]
        # x = self.conv1(x)
        # print(x.size()) # torch.Size([4, 6, 28, 28])
        x = self.pool(F.relu(self.conv1(x)))  # 池化后(2,2)，维度减少了，x = [N, 6, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # x = [N, 16, 5, 5]
        x = x.view(-1, 16 * 5 * 5)  # x = [N, 16 * 5 * 5]
        x = F.relu(self.fc1(x))  # x = [N, 120]
        x = F.relu(self.fc2(x))  # x = [N, 84]
        x = self.fc3(x)  # x = [N, 10]
        return x


net = Net()
criterion = nn.CrossEntropyLoss()  # 交叉熵在单分类问题上基本是标配的方法
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 送入设定的设备中
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
criterion = criterion.to(device)

classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    """
    将转成tensor的图像再变回img
    :param img:
    :return:
    """
    img = img / 2 + 0.5  # unnormalize，由[-1,1]恢复为[0,1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # np.transpose转置，维数转换，将形状为[C,H,W]再变为[H,W,C]，即原图像的格式
    plt.show()


def train_epoch(epoch, iterator):
    running_loss = 0.0
    net.train()
    for i, (inputs,labels) in enumerate(iterator, 0):
        # get the inputs
        # inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)  # outputs是image对应的tensor
        loss = criterion(outputs, labels)  # loss在这里是个标量
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()  # 把前2000个loss值(标量)都加起来
        # 由于统一返回值，tensor 返回都为 tensor , 为了获得 python number 现在需要通过.item()来实现，
        # 考虑到之前的 loss 累加为 total_loss +=loss.data[0], 由于现在 loss 为0维张量,
        # 0维（也就是标量）检索是没有意义的，所以应该使用 total_loss+=loss.item()，
        # 通过.item() 从张量中获得 python number.
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


def evalute_epoch(epoch, iterator):
    correct = 0
    total = 0
    # eval()模式,关闭dropout和batch normalization
    net.eval()
    with torch.no_grad():
        for inputs, labels in iterator:
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('[%d, %d %%] acc for 10000 test images'
          % (epoch, 100 * correct / total))


def main():
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    # net = Net()

    # 使用已经存在的数据导入已有数据集
    transform = transforms.Compose(  # 一起组合几个变换
        [transforms.ToTensor(),
         # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
         # B*C*H*W：batchsize*channel*height*width
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transforms.Normalize这里是针对彩色图像，所以有三个通道（红绿蓝），第一组是三个通道的平均值，第二组是三个通道的标准差
    # https://discuss.pytorch.org/t/understanding-transform-normalize/21730

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    # shuffle=True表示每轮epoch将数据重新洗牌

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)



    # train
    n_epoches = 2
    for epoch in range(n_epoches):
        train_epoch(epoch, trainloader)
        evalute_epoch(epoch, testloader)
    # print('Finished Training')

    ## test
    # 没有验证集，直接

    # 判断训练的结果在哪些内容上比较好
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))



if __name__ == '__main__':
    main()

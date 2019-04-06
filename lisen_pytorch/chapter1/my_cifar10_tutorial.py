# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


if __name__ == '__main__':
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    # shuffle=True表示每轮epoch将数据重新洗牌

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(trainloader)  # trainloader加入迭代器
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))  # 制作一个图像网格,show
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    criterion = nn.CrossEntropyLoss()  # 交叉熵在单分类问题上基本是标配的方法
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0 # loss值初始化为0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

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
    print('Finished Training')





    ## test
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

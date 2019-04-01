import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

"""
数据
"""

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = DataLoader(trainset,
                         batch_size=4,
                         shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = DataLoader(testset,
                        batch_size=4,
                        shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""
模型
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = [N, 3, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))
        # x = [N, 6, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))
        # x = [N, 16, 5, 5]
        x = x.view(-1, 16 * 5 * 5)
        # x = [N, 16 * 5 * 5]
        x = F.relu(self.fc1(x))
        # x = [N, 120]
        x = F.relu(self.fc2(x))
        # x = [N, 84]
        x = self.fc3(x)
        # x = [N, 10]
        return x


"""
训练
"""

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

n_epoches = 15


def train_epoch(epoch):
    running_loss = 0.0
    net.train()
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f'
                  % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


def test_epoch_whole(epoch):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            _, max_idxs = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (max_idxs == labels).sum().item()
    print('[%d, %d %%] acc for 10000 test images'
          % (epoch, 100 * correct / total))


# ???
# def test_epoch_each_class(iterator, epoch):
#     class_correct = list(0. for i in range(10))
#     class_total = list(0. for i in range(10))
#     with torch.no_grad():
#         for inputs, labels in iterator:
#             outputs = net(inputs)
#             _, predicted = torch.max(outputs, 1)
#             c = (predicted == labels).squeeze()
#             for i in range(4):
#                 label =


for epoch in range(n_epoches):
    train_epoch(epoch)
    test_epoch_whole(epoch)
#
#     running_loss = 0.0
#     net.train()
#     for i, (inputs, labels) in enumerate(trainloader, 0):
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         if i % 2000 == 1999:
#             print('[%d, %5d] loss: %.3f'
#                   % (epoch+1, i+1, running_loss / 2000))
#             running_loss = 0.0
#
# correct = 0
# total = 0
# net.eval()
# with torch.no_grad():
#     for inputs, labels in testloader:
#         outputs = net(inputs)
#         _, max_idxs = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (max_idxs == labels).sum().item()
# print('[%d, %d %%] acc for 10000 test images'
#       % (0, 100 * correct / total))

# -*- coding: utf-8 -*-

"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn

torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 200), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1,15) # 输入层
        self.fc2 = nn.Linear(15,30)
        self.fc3 = nn.Linear(30,1) # 输出层


    def forward(self, x):
        x = F.relu(self.fc1(x))      
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

net = Net()     # define the network
net.cuda() 
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
criterion = torch.nn.MSELoss()  # this is for regression mean squared loss
plt.ion()   # something about plotting

for t in range(200):
    b_x = x.cuda()
    b_y = y.cuda()

    prediction = net(b_x)     # input x and predict based on x
    loss = criterion(prediction, b_y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    loss_value = loss.data.cpu().numpy() # 打印loss的值
    #使用.cpu().numpy()的方式将gpu中的数值挪到cpu中，并打印出来，直接在GPU中是无法打印的 

    if t % 5 ==0:
        # print(loss) # 打印出的格式：tensor(0.0039, device='cuda:0', grad_fn=<MseLossBackward>)
        print("Loss = %.4f" % loss_value)
        plt.cla()
        plt.scatter(x.data.cpu().numpy(), y.data.cpu().numpy())
        plt.plot(x.data.cpu().numpy(), prediction.data.cpu().numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.cpu().numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()


# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable  # 包装数据
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # unsqueeze把1维度的变成2维度的
# torch中的数据是有维度的
y = x.pow(2) + 0.2 * torch.rand(x.size())
x, y = Variable(x), Variable(y)


# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_features=1, n_hidden=10, n_output=1)
print(net)
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()


plt.ion()  # something about plotting 设置实时打印的过程
plt.show()



for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)  # prediction在前，真值y在后

    optimizer.zero_grad()  # 每次更新后，梯度会保留在Net中，所以先把梯度置零
    loss.backward()  #
    optimizer.step()

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

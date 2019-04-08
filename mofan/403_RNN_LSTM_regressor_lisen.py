"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
matplotlib
numpy
"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)  # reproducible 种子设置的意义是让结果可以重现

# 超参数
TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

# 模拟数据并绘制正弦曲线
steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)  # float32 for converting torch FloatTensor
# print(steps)
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()


class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_n, h_c):
        if h_n is None and h_c is None:
            h = None
        else:
            h = (h_n, h_c)
        r_out, (h_n,h_c) = self.rnn(x, h)

        r_out = r_out.view(-1, 32)
        outs = self.out(r_out)
        outs = outs.view(-1, TIME_STEP, 1)
        return outs, h_n,h_c


rnn = MyRNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

# h_state = None
h_n =None
h_c = None

plt.figure(1, figsize=(12, 5))
plt.ion()

for step in range(100):
    start, end = step * np.pi, (step + 1) * np.pi  # time range

    # 使用sin预测cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32,
                        endpoint=False)  # float32 for converting torch FloatTensor
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape(batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_n,h_c = rnn(x, h_n, h_c)  # rnn output

    
    h_n = torch.autograd.Variable(h_n.data)
    h_c = torch.autograd.Variable(h_c.data)
    # h_state = torch.autograd.Variable(h_state.data)  # repack the hidden state, break the connection from last iteration
    # 下一次用h_state的问题

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 绘图
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()

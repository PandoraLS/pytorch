"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.1.11
"""
import torch
import torch.utils.data as Data

# torch.manual_seed(1)  # reproducible
BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)  # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)  # this is y data (torch tensor)

torch_dataset = Data.TensorDataset(x, y) # 将数据放入

# 用loader讲训练变成分批次的
loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=False,  # random shuffle for training
    num_workers=2  # subprocesses for loading data
)


def show_batch():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            # train data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()

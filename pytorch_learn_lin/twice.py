import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pprint import pprint


"""
1. 数据集
"""


class SquareDataset(Dataset):
    def __init__(self, samples_num=10000):
        self.samples_num = samples_num
        self.X = [0.001 * i for i in range(samples_num)]
        self.Y = [0.001 * 2 * i for i in range(samples_num)]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.samples_num


square_dataset = SquareDataset(samples_num=10000)
training_iterator = DataLoader(square_dataset, batch_size=10, shuffle=True)

"""
2. 模型
"""


class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 3)
        self.fc2 = nn.Linear(3, 2)
        self.fc3 = nn.Linear(2, output_dim)

    def forward(self, x):
        # x = [batch size]
        x = x.unsqueeze(1)
        # x = [batch size, 1]
        out = self.fc1(x)
        # out = [batch size, 3]
        out = self.fc2(out)
        # out = [batch size, 2]
        out = self.fc3(out)
        # out = [batch size, 1]
        return out


model = NeuralNet(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

print(model)

"""
3. 训练
"""


def train_epoch(iterator):
    model.train()
    last_loss = 0
    for i, (x, y) in enumerate(iterator):
        optimizer.zero_grad()
        y_hat = model(x.float())  # process
        loss = criterion(y_hat, y.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    return last_loss


n_epochs = 10
for epoch in range(n_epochs):
    train_loss = train_epoch(training_iterator)
    print(epoch, 'epoch', train_loss)
    torch.save(model.state_dict(), 'state_dict')
    # if epoch + 1 == n_epochs:
    #     pprint(model.state_dict())

x_test = torch.tensor([3.0])
y_test = model(x_test)
print(x_test, y_test)

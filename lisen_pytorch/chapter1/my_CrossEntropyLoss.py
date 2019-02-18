# -*- coding: utf-8 -*-
# 参考https://medium.com/@aiii/pytorch-e64c248ab428
# 研究nn.CrossEntropyLoss的工作机制
import torch
import torch.nn as nn
from torch.autograd import Variable

output = Variable(torch.rand(1,10))
print(output)
target = Variable(torch.LongTensor([1]))
print(target)

criterion = nn.CrossEntropyLoss()
print(criterion)
loss = criterion(output, target)
print(loss)
print("-----------------")
m = nn.LogSoftmax()
print(m)
input2 = torch.randn(2,3)
print(input2)
output2 = m(input2)
print(output2)
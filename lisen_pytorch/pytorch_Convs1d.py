# -*- coding: utf-8 -*-

"""
https://medium.com/@santi.pdp/how-pytorch-transposed-convs1d-work-a7adac63c4a5
pytorch-convs1d的工作机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

print("1-----------------------------------")
x = torch.ones(1, 1, 7)
conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=2, bias=False)
conv.weight.data = torch.ones(1, 1, 3)
y = conv(x)
print(y)

print("2-----------------------------------")
y = torch.ones(1, 1, 1)
deconv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
deconv.weight.data = torch.ones(1, 1, 3)
x = deconv(y)
print(x)

print("3-----------------------------------")
y = torch.ones(1, 1, 2)
print(y)
x = deconv(y)
print(x)

print("4-----------------------------------")
y = torch.ones(1, 1, 2)
deconv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=4, padding=0, bias=False)
deconv.weight.data = torch.ones(1, 1, 3)
x = deconv(y)
print(x)

print("5-----------------------------------")
y = torch.ones(1, 1, 2)
deconv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=4, stride=4, padding=0, bias=False)
deconv.weight.data = torch.ones(1, 1, 4)
x = deconv(y)
print(x)

print("6-----------------------------------")
y = torch.ones(1, 1, 2)
deconv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=8, stride=4, padding=0, bias=False)
deconv.weight.data = torch.ones(1, 1, 8)
x = deconv(y)
print(x.size())
print(x)

print("7-----------------------------------")
# y = torch.ones(1,1,2)
deconv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=8, stride=4, padding=2, bias=False)
deconv.weight.data = torch.ones(1, 1, 8)
x = deconv(y)
# print(x.size())
print(x)

print("8-----------------------------------")
# y = torch.ones(1,1,2)
deconv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=8, stride=4, padding=0, output_padding=1,
                            bias=False)
deconv.weight.data = torch.ones(1, 1, 8)
x = deconv(y)
# print(x.size())
print(x)


print("9-----------------------------------")
# y = torch.ones(1,1,2)
deconv = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=8, stride=4, padding=2, output_padding=1,
                            bias=False)
deconv.weight.data = torch.ones(1, 1, 8)
x = deconv(y)
# print(x.size())
print(x)

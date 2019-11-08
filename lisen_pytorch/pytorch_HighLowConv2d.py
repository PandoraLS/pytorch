# -*- coding: utf-8 -*-
# @Time    : 2019/11/8 10:58
# @Author  : Li Sen
# pytorch
import torch

a = torch.ones([4, 8])
b = torch.zeros([4, 8])
c = torch.cat([a, b], 0)  # 第0个维度stack
d1, d2 = torch.chunk(c, 2, 0)
print("1--------------------------------")
print('a.size: ',end='');print(a.size())
print(a)
print('b.size: ',end='');print(b.size())
print(b)
print('c.size: ',end='');print(c.size())
print(c)
print('d1.size: ',end='');print(d1.size())
print(d1)
print('d2.size: ',end='');print(d2.size())
print(d2)

print('2----------------------------------')
a2 = a.view(1, 1, a.size(0), a.size(1))
print('a2.size: ',end='');print(a2.size())
print(a2)
b2 = b.view(1,1,b.size(0),b.size(1))
print('b2.size: ',end='');print(b2.size())
print(b2)
c2 = torch.cat([a2, b2], 2)  # 第0个维度stack
print('c2.size: ',end='');print(c2.size())
print(c2)
dd1,dd2 = torch.chunk(c2,2,2)

print('3------------------------------------')
print('dd1.size: ',end='');print(dd1.size())
print(dd1)
print('dd2.size: ',end='');print(dd2.size())
print(dd2)
c22 = torch.cat([dd1,dd2],2)
print('c22.size: ',end='');print(c22.size())
print(c22)

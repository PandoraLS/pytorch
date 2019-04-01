import os
import json
from pprint import pprint
import torch


state_dict = torch.load('state_dict')
for k, v in state_dict.items():
    print(k)
    print(v.size())
    print(v)
    print('-' * 30)
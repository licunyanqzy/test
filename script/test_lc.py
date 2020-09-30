import torch
import numpy as np

a = torch.tensor([1.0, 2]).cuda()
b = torch.tensor([4.0, 6]).cuda()
c = torch.norm(a - b)
print(c)

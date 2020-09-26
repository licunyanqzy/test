import torch
import numpy as np

a = torch.tensor([3, 4])
b = np.linalg.norm(a)
print(b)
c = torch.zeros(2, 3)
c[1, 1] = b
print(c)
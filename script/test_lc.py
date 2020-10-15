import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pylab
import torch.nn.functional as F
import multiprocessing
import timeit
import operator


a = np.arange(1, 11).reshape([2, 5])
b = np.cumsum(a)
print(a)
print(b)


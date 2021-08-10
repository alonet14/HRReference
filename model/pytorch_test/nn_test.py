import torch
from torch import nn
from torch.nn import functional as F

net=nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X=torch.rand(2, 20)
net(X)

import torch.nn as nn

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms



class lstmNet(nn.Module):
    def __init__(self):
        self.name = "large"

    def forward(self, x):

        return x
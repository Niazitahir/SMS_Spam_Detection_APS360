import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import pandas as pd
import csv

totalData = []

with open('RawData/sms+spam+collection/SMSSpamCollection', newline = '') as csvfile:
    spamreader = csv.reader(csvfile, quotechar='|')
    for row in spamreader:
        #print(', '.join(row))
        totalData.append(row[0])
#print(totalData)
labels = []
values = []
for row in totalData:
    labels.append(row.split()[0]) 


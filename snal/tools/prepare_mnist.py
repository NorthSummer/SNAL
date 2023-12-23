import os
# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt 
# torch.manual_seed(1)  # reproducible
DOWNLOAD_MNIST = False
  
# Mnist digits dataset
if not(os.path.exists('/home/liuminzhe/ljn/pytorch-cifar/data/mnist')) or not os.listdir('/home/liuminzhe/ljn/pytorch-cifar/data/mnist/'):
  # not mnist dir or mnist is empyt dir
  DOWNLOAD_MNIST = True
  
train_data = torchvision.datasets.MNIST(
  root='/home/liuminzhe/ljn/pytorch-cifar/data/mnist/',
  train=True,                   # this is training data
  transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
                          # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
  download=DOWNLOAD_MNIST,
)
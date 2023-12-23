
import torch
import spikingjelly
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, surrogate, functional
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm

import os
import argparse


#class auto_encoder(nn.Module):
import shutil
import sys
import os
import torch
import random

from .strategy import Strategy

from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import cv2
from PIL import Image
from utils import *
import PIL
import sys
import math


from models.vgg import VGG

import torch.nn.functional as F
import random
from tqdm import tqdm



class coreset_sample(Strategy):
    def __init__(self, set, net, cfg, dataset, transforms):
        super(max_entropy_sample, self).__init__(set, net, cfg, dataset, transforms)
        weight_dir = cfg["model"][dataset]["weight_dir"]["vgg16"]
        net.load_state_dict(torch.load(weight_dir), False)
        net.cuda()
        
        
        self.net = net
        self.sample = []
    
    def query(self, amount):
        pass
        
        
        
    def update(self):
        f0 = open(self.label_path, 'a', encoding = 'utf-8')
        f1 = open(self.all_path, 'r', encoding = 'utf-8')
        f2 = open(self.label_path, 'r', encoding = 'utf-8')
        
        f3 = open(self.unlabel_path, 'a')
        
        list1 = list(f1.readlines())
        list2 = list(f2.readlines())
        for l1 in list1:
            l1 = l1[:-2]
        for l2 in list1:
            l2 = l2[:-2]
            

        for aline in list1:
            if aline not in list2:
                f3.write(aline)  
                
        for s in self.sample:
            
            f0.write(s)
            f0.write(' ')
            f0.write(self.A[s])
        
        
        
               
    
        
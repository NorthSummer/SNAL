import shutil
import sys
import os
import torch
import random

from .strategy import Strategy

from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import cv2
from PIL import Image
from utils import *
import PIL
import sys
import math
import numpy

from models.vgg import VGG

import torch.nn.functional as F
import random
from tqdm import tqdm


def maxminnorm(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t



class k_sample(Strategy):
    def __init__(self, set, net, cfg, dataset, transforms):
        super(k_sample, self).__init__(set, net, cfg, dataset, transforms)
        weight_dir = cfg["model"][dataset]["weight_dir"]["vgg16"]
        
        weight_dir = "/home/liuminzhe/ljn/pytorch-cifar/checkpoint/ckpt_cifar10_5000.pth"
        
        net.load_state_dict(torch.load(weight_dir), False)
        net.cuda()
        
        self.vgg = models.vgg16(pretrained = True).cuda()
        self.net = net
        self.sample = []
        
    def get_feature(self, x, layer_index):
          features = self.vgg.features
          for index, layer in enumerate(features):
              x = layer(x)
              if index == layer_index:
                  break
              else:
                  continue
          
          return x 
          
    def to_image(self, x):
        x = Variable(torch.squeeze(x, dim =0).float(), requires_grad = False)
        x = Variable(torch.squeeze(x, dim =0).float(), requires_grad = False)
        
        x = x.detach().cpu()
        x = numpy.array(x)
        
        x= maxminnorm(x)
        
        x=np.round(x*255)
        cv2.imwrite('./img.jpg',x)
        #print(x)
        
        
    
    def query(self, amount):
        ents = {}
     
        transforms_ = transform_test_cifar10 = transforms.Compose([
          transforms.ToTensor(),
            ])
        print('==>querying')
        unlabel_list =  list(self.U.keys())
        n = 0
        for unlabel in tqdm(unlabel_list):
            img = Image.open(unlabel).convert("RGB")
            #to_tensor = transforms_(img)
            img = np.array(img, dtype = np.float32)
            img = np.transpose(img, (2,0,1))
            to_tensor = torch.from_numpy(img)
            to_tensor = Variable(torch.unsqueeze(to_tensor, dim = 0).float(), requires_grad = False).cuda()
            #prob = self.net.predict_prob(to_tensor)  
            #features = self.net.extract_feature(to_tensor)
            
            features = self.vgg.features
            
            #f1 = features[9]
            #f2 = features[12]
            
            f1 = self.get_feature(to_tensor, 9)
            f2 = self.get_feature(to_tensor, 12)
            print(f1)
            
            #print(f1.shape)
            f = torch.cat((f1, f2), 1)
            #print(f)
            
            #l = torch.nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 1, padding = 1, bias = False)
            weight = [[1,]]
            weight = torch.FloatTensor(weight).expand(1, 256, 1, 1).cuda()
            
            kernel = nn.Parameter(data = weight, requires_grad = False)
            x = torch.nn.functional.conv2d(f, weight = kernel)
            #print(x)
            x = self.to_image(x)
            
            #print(x.shape)
            #l = nn.Sequential(x)
            #l = l.cuda()
            #out = x(f)
            #print(unlabel)
            #print(x)
            
            print(unlabel)
            n = n + 1
            if n ==1:
                break
            
        return None
           
           
           
           
    '''       
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
    '''        
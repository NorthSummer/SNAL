import shutil
import sys
import os
import torch
import random



from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import cv2
from PIL import Image
from utils import *
import PIL
import sys

from models.vgg import VGG

import torch.nn.functional as F


class Strategy():
    def __init__(self, set, net, cfg, dataset, transform):
        self.net = net
        self.set = set
        self.all_path = cfg["dataset"][dataset]["all_path"]
        self.label_path = cfg["dataset"][dataset]["label_path"]
        self.unlabel_path = cfg["dataset"][dataset]["unlabel_path"]
                
        self.A, self.all_amount = self.from_labels(self.all_path)
        self.U, self.unlabel_amount = self.from_labels(self.unlabel_path)
        
        self.transforms = transforms
        #self.transforms = transforms.Compose([transforms.ToTensor(),
         #                 transforms.Normalize((0.5088964127604166, 0.48739301317401956, 0.44194221124387256),(0.2682515741720801, 0.2573637364478126, 0.2770957707973042)),
          #                ])
        if dataset == "cifar10":
            self.transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            
        elif dataset == "cifar100":
            self.transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),])
            
        elif dataset == "caltech101":
            self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(( 0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
                    
        elif dataset == "caltech256":
            self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(( 0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        


                             
                
        
        
    
    def query(self, amount):
        pass
        
    def predict_prob(self):
        pass  
        
    def update(self):      
        pass
        
    def from_labels(self, dir_):
        dic = {}
        f = open(dir_, 'r')
        amount = 0
        for line in f.readlines():
            data = line.split(' ')
            dic[data[0]] = data[1]
            amount = amount + 1
            
        return dic, amount
        
        
    
    
        
    
        
        
        
        
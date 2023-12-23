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

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

class bayesian_sample(Strategy):
    def __init__(self, set, net, cfg, dataset, transform):
        super(bayesian_sample, self).__init__(set, net, cfg, dataset, transform)
        weight_dir = cfg["model"][dataset]["weight_dir"]["resnet34"]
        net.load_state_dict(torch.load(weight_dir), False)
        #net = net.module
        net.to(device)
        
        self.num_class = net.num_classes
        self.net = net
        self.sample = []
        
    
    def query(self, amount, dropout_i):
        bys_KL = {}
    
        print('==>querying')
        unlabel_list =  list(self.U.keys())
        for unlabel in tqdm(unlabel_list):
            img = Image.open(unlabel).convert("RGB")
            to_tensor = self.transforms(img)
            to_tensor = Variable(torch.unsqueeze(to_tensor, dim = 0).float(), requires_grad = False).to(device)
            #prob = self.net.predict_prob(to_tensor)  
            
            all_score = np.zeros(self.num_class)#modify classes here
            ent_all = 0
            
            for itr in range(dropout_i-1):
                
                vector = self.net.predict_prob_dropout(to_tensor)
                vector = vector.cpu().detach()
                vector = np.array(vector, dtype = np.float32)
                #print(vector)
                vector = vector[0]
                #print(vector)
                #print(all_score)
                all_score = np.add(all_score, vector)
  
   
                ent_sum = 0
                for v in vector:
                    v = float(v)
                    ent =  v * math.log(v)
                    ent_sum = ent_sum + ent
                
                ent_all = ent_all + ent_sum
            
            F = ent_all/int(dropout_i)
            
            all_score = np.divide(all_score, dropout_i)
            ent_sum = 0
            #print(all_score)
            for s in all_score:
                    s = float(s)
                    ent =  s * math.log(s)
                    ent_sum = ent_sum + ent
            G = ent_sum
            
            U = G - F
            
            bys_KL[unlabel] = U
        
        ents_list = sorted(bys_KL.items(), key = lambda x:x[1], reverse = False)   
            
        count = 0
        for img, ent in ents_list:
            self.sample.append(img)
            count = count + 1
            if count >= amount:
                break      
       # print(self.sample)
        
        
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
        
        
        
        
        
        
        
        
        
        
        
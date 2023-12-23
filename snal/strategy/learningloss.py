import torch
import torch.nn as nn 

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


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()
        
        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)
    
    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out

            
def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()
      
class learning_loss_sample(Strategy):
    def __init__(self, set, net, loss_net, cfg, dataset, transform):
        super(bayesian_sample, self).__init__(set, net, cfg, dataset, transform)
        weight_dir = cfg["model"][dataset]["weight_dir"]["resnet34"]
        net.load_state_dict(torch.load(weight_dir), False)
        net.cuda() 
        #resnet18    = resnet.ResNet18(num_classes=10).cuda()
        loss_module = LossNet().cuda()
        models = {'backbone': net, 'module': loss_module}
        
        self.num_class = net.num_classes
        self.net = net
        self.models = models
        self.sample = []
        self.loss_net = loss_net
        
        
    
    def query(self, amount):
        LL_uncertainty = {}
        models = {"backbone":self.net, "loss_module":self.loss_net}
        print('==>querying')
        unlabel_list =  list(self.U.keys())
        models["backbone"].eval()
        models["loss_module"].eval()
        
        for unlabel in tqdm(unlabel_list):
            img = Image.open(unlabel).convert("RGB")
            to_tensor = self.transforms(img)
            to_tensor = Variable(torch.unsqueeze(to_tensor, dim = 0).float(), requires_grad = False).cuda() 
            
            scores, features = self.models["backbone"](to_tensor)
            pred_loss = models["loss_module"](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            uncertainty = pred_loss.cpu()
            
            LL_uncertainty[unlabel] = uncertainty
            #all_score = np.zeros(self.num_class)#modify classes here
            #ent_all = 0
            '''
            for itr in range(dropout_i-1):
                
                vector = self.net.predict_prob_dropout(to_tensor)
                vector = vector.cpu().detach()
                vector = np.array(vector, dtype = np.float32)

                vector = vector[0]

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

            for s in all_score:
                    s = float(s)
                    ent =  s * math.log(s)
                    ent_sum = ent_sum + ent
            G = ent_sum
            
            U = G - F
            
            bys_KL[unlabel] = U
            '''
        llu_list = sorted(LL_uncertainty.items(), key = lambda x:x[1], reverse = False)   
            
        count = 0
        for img, ent in llu_list:
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
    
    
    def train_spec(self):
    
        models['backbone'].train()
        models['module'].train()
        global iters
    
        for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
            inputs = data[0].cuda()
            labels = data[1].cuda()
            iters += 1
    
            optimizers['backbone'].zero_grad()
            optimizers['module'].zero_grad()
    
            scores, features = models['backbone'](inputs)
            target_loss = criterion(scores, labels)
    
            if epoch > epoch_loss:
                # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()
            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
    
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            loss            = m_backbone_loss + WEIGHT * m_module_loss
    
            loss.backward()
            optimizers['backbone'].step()
            optimizers['module'].step()
    
        
        
        
        
    
    
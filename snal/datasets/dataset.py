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


from skimage.color import gray2rgb

def dict_labels(dir_):
    amount = 0
    labels = {}
    f = open(dir_, 'r')
    for line in f.readlines():
        data = line.split(' ')
        labels[data[0]] = data[1]
        amount = amount + 1
    return amount, labels
        




class cifar10_train(Dataset):
    def __init__(self, cfg, mode, transform):        
        self.cfg = cfg
        
        self.mode = mode
        
        self.transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
        #self.transforms = transforms.Compose([
        #    transforms.ToTensor(), ])
        self.label_path = self.cfg["dataset"]["cifar10"]["label_path"]
        self.all_path = self.cfg["dataset"]["cifar10"]["all_path"]
        self.unlabel_path = self.cfg["dataset"]["cifar10"]["unlabel_path"]
        
        _, all_labels = dict_labels(self.all_path)
        l_amount, self.L = dict_labels(self.label_path)
        #print(self.L.keys())
        u_amount, self.U = dict_labels(self.unlabel_path)
        
    
    def __getitem__(self, index):
        l_label_path = list(self.L.keys())[index]
        l_data = Image.open(l_label_path).convert("RGB")
        l_data = self.transforms(l_data)
        
        l_target = self.L[l_label_path]
        l_target = torch.tensor(np.array(l_target, dtype = np.float32)).long()
        
        return l_data, l_target
        
    def __len__(self):
        if self.mode=="label":
            return len(self.L.keys())
        elif self.mode == "unlabel":
            return len(self.U.keys())
        else:
            print("please enter mode 'label' or 'unlabel'")
        

        
class cifar10_test(Dataset):
    def __init__(self, cfg, transform):        
        self.cfg = cfg
        
        self.transform = transforms.Compose([
                  transforms.RandomCrop(32, padding=4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              ])
        
        self.label_path = self.cfg["dataset"]["cifar10"]["label_path"]
        self.all_path = self.cfg["dataset"]["cifar10"]["all_path"]
        self.unlabel_path = self.cfg["dataset"]["cifar10"]["unlabel_path"]
        self.test_path = self.cfg["dataset"]["cifar10"]["test_path"]
        
        _, all_labels = dict_labels(self.all_path)
        l_amount, self.L = dict_labels(self.test_path)
        u_amount, self.U = dict_labels(self.unlabel_path)
        
    
    def __getitem__(self, index):
        l_label_path = list(self.L.keys())[index]
        l_data = Image.open(l_label_path).convert("RGB")
        l_data = self.transform(l_data)
        
        l_target = self.L[l_label_path]
        l_target = torch.tensor(np.array(l_target, dtype = np.float32)).long()
        
        return l_data, l_target
        
    def __len__(self):
        
        return len(self.L.keys())
 
 
class cifar100_train(Dataset):
    def __init__(self, cfg, mode, transform):        
        self.cfg = cfg
        
        self.mode = mode
        self.transforms = transforms.Compose([
                 transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
            ])
        
        self.label_path = self.cfg["dataset"]["cifar100"]["label_path"]
        self.all_path = self.cfg["dataset"]["cifar100"]["all_path"]
        self.unlabel_path = self.cfg["dataset"]["cifar100"]["unlabel_path"]
        
        _, all_labels = dict_labels(self.all_path)
        l_amount, self.L = dict_labels(self.label_path)
        #print(self.L.keys())
        u_amount, self.U = dict_labels(self.unlabel_path)
        
    
    def __getitem__(self, index):
        l_label_path = list(self.L.keys())[index]
        l_data = Image.open(l_label_path).convert("RGB")
        l_data = self.transforms(l_data)
        
        l_target = self.L[l_label_path]
        l_target = torch.tensor(np.array(l_target, dtype = np.float32)).long()
        
        return l_data, l_target
        
    def __len__(self):
        if self.mode=="label":
            return len(self.L.keys())
        elif self.mode == "unlabel":
            return len(self.U.keys())
        else:
            print("please enter mode 'label' or 'unlabel'")
        

        
class cifar100_test(Dataset):
    def __init__(self, cfg, transform):        
        self.cfg = cfg
        
        self.transform = transforms.Compose([
                  transforms.RandomCrop(32, padding=4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),(0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
              ])
        
        self.label_path = self.cfg["dataset"]["cifar100"]["label_path"]
        self.all_path = self.cfg["dataset"]["cifar100"]["all_path"]
        self.unlabel_path = self.cfg["dataset"]["cifar100"]["unlabel_path"]
        self.test_path = self.cfg["dataset"]["cifar100"]["test_path"]
        
        _, all_labels = dict_labels(self.all_path)
        l_amount, self.L = dict_labels(self.test_path)
        u_amount, self.U = dict_labels(self.unlabel_path)
        
    
    def __getitem__(self, index):
        l_label_path = list(self.L.keys())[index]
        l_data = Image.open(l_label_path).convert("RGB")
        l_data = self.transform(l_data)
        
        l_target = self.L[l_label_path]
        l_target = torch.tensor(np.array(l_target, dtype = np.float32)).long()
        
        return l_data, l_target
        
    def __len__(self):
        
        return len(self.L.keys()) 
 
 
 
   
  
'''
class cifar100_train(Dataset):
    def __init__(self, args, mode, transform):
        #self.img_dir = '/home/'
        self.mode = mode
        self.all_path = args.all_path
        self.train_label_path = args.label_path
        self.unlabel_path = args.unlabel_path
        self.args = args
        self.transform = transform
        
        _, all_labels = dict_labels(self.all_path)
        L_amount, l_labels = dict_labels(self.train_label_path)
       
        
        self.L = l_labels
       
        
        
    def __getitem__(self, index):
        
        l_data_path = list(self.L.keys())[index]
        l_data = Image.open(l_data_path).convert("RGB")
        
        l_target = self.L[l_data_path]
        l_target = torch.tensor(np.array(l_target, dtype = np.float32)).long()
        
        l_data = self.transform(l_data)
       
        
        
        return l_data, l_target#, u_data
        
    def __len__(self):
        if self.mode == 'label':
            return len(self.L.keys())
        elif self.mode == 'unlabel':
            return len(self.U.keys())   
        else:
            print("please enter mode 'label' or 'unlabel'")         
        
        
        
class cifar100_test(Dataset):
    def __init__(self, args, transform):
        #self.img_dir = '/home/'
        
        self.all_path = args.all_path
        self.test_label_path = args.test_path
        self.transform = transform
        
        _, all_labels = dict_labels(self.all_path)
        L_amount, l_labels = dict_labels(self.test_label_path)
        
        self.L = l_labels
        
        
    def __getitem__(self, index):
        
        l_data_path = list(self.L.keys())[index]
        l_data = Image.open(l_data_path).convert("RGB")
        #l_data = np.array(Image.open(l_data_path).convert('RGB'), dtype = np.float32)        
        #l_data = np.transpose(l_data, (2,0,1))
        l_target = self.L[l_data_path]
        l_target = torch.tensor(np.array(l_target, dtype = np.float32)).long()
        
        l_data = self.transform(l_data)
       
        
        
        return l_data, l_target
        
    def __len__(self):
        
        return len(self.L.keys())
                  
'''         
        
class mnist_train(Dataset):
    def __init__(self, cfg, mode, transform):        
        self.cfg = cfg
        
        self.mode = mode
        self.transforms = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,)),
                          ])
        
        self.label_path = self.cfg["dataset"]["mnist"]["label_path"]
        self.all_path = self.cfg["dataset"]["mnist"]["all_path"]
        self.unlabel_path = self.cfg["dataset"]["mnist"]["unlabel_path"]
        
        _, all_labels = dict_labels(self.all_path)
        l_amount, self.L = dict_labels(self.label_path)
        u_amount, self.U = dict_labels(self.unlabel_path)
        
    
    def __getitem__(self, index):
        l_label_path = list(self.L.keys())[index]
        l_data = Image.open(l_label_path).convert("RGB")
        l_data = self.transforms(l_data)
        
        l_target = self.L[l_label_path]
        l_target = torch.tensor(np.array(l_target, dtype = np.float32)).long()
        
        return l_data, l_target
        
    def __len__(self):
        if self.mode=="label":
            return len(self.L.keys())
        elif self.mode == "unlabel":
            return len(self.U.keys())
        else:
            print("please enter mode 'label' or 'unlabel'")
        

        
class mnist_test(Dataset):
    def __init__(self, cfg, transform):        
        self.cfg = cfg
        
        self.transforms = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,)),
                          ])
        
        
        self.test_path = self.cfg["dataset"]["mnist"]["test_path"]
        
        
        l_amount, self.L = dict_labels(self.test_path)
       
        
    
    def __getitem__(self, index):
        l_label_path = list(self.L.keys())[index]
        l_data = Image.open(l_label_path).convert("RGB")
        l_data = self.transforms(l_data)
        
        l_target = self.L[l_label_path]
        l_target = torch.tensor(np.array(l_target, dtype = np.float32)).long()
        
        return l_data, l_target
        
    def __len__(self):
        
        return len(self.L.keys())
              
        
        
class caltech101_train(Dataset):
    def __init__(self, cfg, mode, transform):        
        self.cfg = cfg
        
        self.mode = mode
        self.transforms = transforms.Compose([
                          transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize(( 0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                          ])
        
        self.label_path = self.cfg["dataset"]["caltech101"]["label_path"]
        self.all_path = self.cfg["dataset"]["caltech101"]["all_path"]
        self.unlabel_path = self.cfg["dataset"]["caltech101"]["unlabel_path"]
        
        _, all_labels = dict_labels(self.all_path)
        l_amount, self.L = dict_labels(self.label_path)
        u_amount, self.U = dict_labels(self.unlabel_path)
        
    
    def __getitem__(self, index):
        l_label_path = list(self.L.keys())[index]
        l_data = Image.open(l_label_path).convert("RGB")
        l_data = self.transforms(l_data)
        
        l_target = self.L[l_label_path]
        l_target = torch.tensor(np.array(l_target, dtype = np.float32)).long()
        
        return l_data, l_target
        
    def __len__(self):
        if self.mode=="label":
            return len(self.L.keys())
        elif self.mode == "unlabel":
            return len(self.U.keys())
        else:
            print("please enter mode 'label' or 'unlabel'")
        

        
class caltech101_test(Dataset):
    def __init__(self, cfg, transform):        
        self.cfg = cfg
        
        self.transforms = transforms.Compose([
                          transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize(( 0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                          ])
        
       
        self.test_path = self.cfg["dataset"]["caltech101"]["test_path"]
        
       
        l_amount, self.L = dict_labels(self.test_path)
        
        
    
    def __getitem__(self, index):
        l_label_path = list(self.L.keys())[index]
        l_data = Image.open(l_label_path).convert("RGB")
        l_data = self.transforms(l_data)
        
        l_target = self.L[l_label_path]
        l_target = torch.tensor(np.array(l_target, dtype = np.float32)).long()
        
        return l_data, l_target
        
    def __len__(self):
        
        return len(self.L.keys())
                      
                


class caltech256_train(Dataset):
    def __init__(self, cfg, mode, transform):        
        self.cfg = cfg
        
        self.mode = mode
        self.transforms = transforms.Compose([
                          transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize(( 0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                          ])
        
        self.label_path = self.cfg["dataset"]["caltech256"]["label_path"]
        self.all_path = self.cfg["dataset"]["caltech256"]["all_path"]
        self.unlabel_path = self.cfg["dataset"]["caltech256"]["unlabel_path"]
        
        _, all_labels = dict_labels(self.all_path)
        l_amount, self.L = dict_labels(self.label_path)
        u_amount, self.U = dict_labels(self.unlabel_path)
        
    
    def __getitem__(self, index):
        l_label_path = list(self.L.keys())[index]
        l_data = Image.open(l_label_path).convert("RGB")
        l_data = self.transforms(l_data)
        
        l_target = self.L[l_label_path]
        l_target = torch.tensor(np.array(l_target, dtype = np.float32)).long()
        
        return l_data, l_target
        
    def __len__(self):
        if self.mode=="label":
            return len(self.L.keys())
        elif self.mode == "unlabel":
            return len(self.U.keys())
        else:
            print("please enter mode 'label' or 'unlabel'")
        

        
class caltech256_test(Dataset):
    def __init__(self, cfg, transform):        
        self.cfg = cfg
        
        self.transforms = transforms.Compose([
                          transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize(( 0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                          ])
        
       
        self.test_path = self.cfg["dataset"]["caltech256"]["test_path"]
        
       
        l_amount, self.L = dict_labels(self.test_path)
        
        
    
    def __getitem__(self, index):
        l_label_path = list(self.L.keys())[index]
        l_data = Image.open(l_label_path).convert("RGB")
        l_data = self.transforms(l_data)
        
        l_target = self.L[l_label_path]
        l_target = torch.tensor(np.array(l_target, dtype = np.float32)).long()
        
        return l_data, l_target
        
    def __len__(self):
        
        return len(self.L.keys())                
        
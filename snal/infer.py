import torch


import random
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import yaml
from datasets.dataset import cifar10_train, caltech101_train
from tqdm import tqdm

from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from models import *

transform_train_cifar10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



#net = VGG_cifar10('VGG16')
net = VGG_caltech101('VGG16')


net = net.cuda()
ff = open("/home/lijianing/AL/pytorch-cifar/config/cfg.yaml")
cfg = yaml.safe_load(ff)

ROOT = '/home/lijianing/AL/pytorch-cifar/data/caltech101/' 

#set = cifar10_train(cfg = cfg, mode = 'label', transform = transform_train_cifar10)
set = caltech101_train(cfg, mode = 'label', transform = None)

transform = torchvision.transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])                            
                               
transform_101 = transforms.Compose([
                          transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize(( 0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                          ])

    
dataset_inf = set  


loader = DataLoader(dataset_inf, batch_size = 1, shuffle = True, num_workers = 4)


#net = LeNet_cifar10()


net.load_state_dict(torch.load('/home/lijianing/AL/pytorch-cifar/checkpoint/101ckpt.pth'), False)

net = net.cuda() 

                              
                            
def infer_distance(idx1, idx2, dataset, net):
    net = net.cuda()
    
    
    
    data1, label1 = dataset[idx1]
    data2, label2 = dataset[idx2]
    
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    print(label1, label2)
    
    #im1 = np.uint8(np.transpose(data1, (1, 2, 0)))
    im1 = np.transpose(data1, (1, 2, 0))
    #data2 = data2.transpose(1, 2, 0)
    im2 = np.transpose(data2, (1, 2, 0))

    #img1 = Image.fromarray(im1).save('1.png')
    
    plt.subplot(1,2,1)
    plt.imshow(im1)
    plt.subplot(1,2,2)
    plt.imshow(im2)
    
    #plt.savefig('1.jpg')
    
    
 
    
    
    data1 = torch.from_numpy(data1)
    data2 = torch.from_numpy(data2)
    
    
    
    data1 = Variable(torch.unsqueeze(data1, dim=0).float(), requires_grad=False)
    data2 = Variable(torch.unsqueeze(data2, dim=0).float(), requires_grad=False)
    data1 = data1.cuda()
    data2 = data2.cuda()
    
    vec1, vec2 = net.distribute_forward(data1, data2)
    
    
    dis = F.pairwise_distance(vec1, vec2, p=2)
    
    return dis
    
transform = torchvision.transforms.Compose([
                transforms.ToTensor(),
            ])   
    
def main():
    '''
    #f3 = open('/home/lijianing/AL/pytorch-cifar/data/cifar/labels.txt', 'r')
    f3 = open('/home/lijianing/AL/pytorch-cifar/data/caltech101/labels.txt', 'r')
    for line in tqdm(f3.readlines()):
        data = line.split(' ')
        img = Image.open(data[0]).convert("RGB")
        tensor = transform_101(img)   
        tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
        tensor = tensor.cuda()
        tensor = net.forward(tensor)
        tensor_name = data[0].replace(".jpg",".pt")
            
        torch.save(tensor, tensor_name)
    '''
    label_tensor_list = []
    unlabel_tensor_list = []
    
    label_list = list(set.L.keys())
    unlabel_list = list(set.U.keys())
    
    
    for label in label_list:
            #label = label[:-2]
        label = label.replace("jpg","pt")
        label_tensor_list.append(label)
            
    for unlabel in unlabel_list:
            #unlabel = unlabel[:-2]
        unlabel = unlabel.replace("jpg","pt")
        unlabel_tensor_list.append(unlabel)
        
        #label_tensor_list = list(filter(lambda s: s.replace(".jpg",".pt")[:-2], label_list))
        #unlabel_tensor_list = list(filter(lambda s: s.replace(".jpg","pt")[:-2], unlabel_list))
    ch = random.choice(label_tensor_list)
    cht = torch.load(ch)
    print(cht.size())
    cht = F.softmax(cht, dim=1)
    print(ch)
    
    distance = {}
    sort_list = []
      
    for dt in tqdm(unlabel_tensor_list):
        dtt = torch.load(dt)
        dtt = F.softmax(dtt, dim=1)
        dis = F.cosine_similarity(cht, dtt, dim=1)
        distance[dt]=float(dis[0])
        
    #print(distance)
     
    sort_list = sorted(distance.items(), key = lambda x:x[1], reverse = True)
        
    large_dis = []
        
        #print(self.sample)
        
    count = 0
    amount = 100 
    for itm, dis in sort_list:
        if count < amount:
            large_dis.append(itm)
            count = count + 1   
            #print(itm)  
    
    print(large_dis)
   
    
    
    
    
    '''
    for idx1 in range(1200, 5000):
        idx2 = 1021
        
        dis = infer_distance(idx1, idx2, dataset = dataset_inf, net = net)
        
        print(dis, idx1)
        if dis[0] > 0.95:
            print(idx1)

    idx1 = 1578
    idx2 = 1021
    
    dis = infer_distance(idx1, idx2, dataset = dataset_inf, net = net)
    '''
if __name__ == "__main__":
    main()
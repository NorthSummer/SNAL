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
import faiss


from models.vgg import VGG

import torch.nn.functional as F
import random
from tqdm import tqdm

device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

class siamese_sample_hk(Strategy):        # whether make thransforms to the compared data???
    def __init__(self, set, net, cfg, dataset, transform):
        super(siamese_sample_hk,self).__init__(set, net, cfg, dataset, transform) #?
        
        weight_dir = cfg["model"][dataset]["weight_dir"]["resnet34"]
        #net.load_state_dict(torch.load(weight_dir), False)
        net.to(device)
        net.eval()
        
        self.sample = None
        #self.onedis = []
        self.polardis = {}
        
        f1 = open(self.label_path, 'r')
        f2 = open(self.unlabel_path, 'r')
        f3 = open(self.all_path, 'r')
        
        label_tensor_dir = os.path.join(cfg["dataset"][dataset]["tensor_path"], 'labeled')
        
        
        
        print("==>transforming tensor")
        
        
        for line in tqdm(f3.readlines()):
            data = line.split(' ')
            img = Image.open(data[0]).convert("RGB")
            tensor = self.transforms(img)
            tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
            tensor = tensor.to(device)
            tensor = net.forward(tensor)
            #tensor = F.softmax(tensor)  # add to change core set
            tensor_name = data[0].replace(".jpg",".pt")
            #print(tensor_name)
            self.d = len(tensor[0])
            break
            torch.save(tensor, tensor_name)
         
        
    def query(self, amount):
        print("==>querying, first step making comparasion")
        label_list = list(self.set.L.keys())
        unlabel_list = list(self.set.U.keys())
        
        
        
        label_amount = len(label_list)
        to_be_clustered = torch.zeros((label_amount, self.d))
        
        #label_tensor_list = list(filter(lambda s: s.replace(".jpg",".pt")[:-2], label_list))
        #unlabel_tensor_list = list(filter(lambda s: s.replace(".jpg","pt")[:-2], unlabel_list))
        label_tensor_list = []
        unlabel_tensor_list = []
        for idx, label in enumerate(label_list):
            #label = label[:-2]
            label = label.replace("jpg","pt")
            l_tensor = torch.load(label)
            to_be_clustered[idx,:] = l_tensor
            
            label_tensor_list.append(label)
            
        for unlabel in unlabel_list:
            #unlabel = unlabel[:-2]
            unlabel = unlabel.replace("jpg","pt")
            unlabel_tensor_list.append(unlabel)
        
        to_be_clustered = np.array(to_be_clustered.detach().cpu())
        result = run_hkmeans(to_be_clustered)   
        cens = result['centroids']   
        #print(len(cens)) 
        #print(cens[0].size())
    
        #print(label_tensor_list)
        start = -1*amount-1
        for unlabel_tsr in tqdm(unlabel_tensor_list):
            onedis = []
            unlabel_tensor = torch.load(unlabel_tsr).to(device)
            
            for id, label_tsr in enumerate(range(cens[2].size(0))):  #consistence with below
                label_tensor = cens[2][id]
                distance = F.pairwise_distance(unlabel_tensor, label_tensor, p=2)
                onedis.append(float(distance))

                
            min_dis = min(onedis)
            self.polardis[unlabel_tsr.replace(".pt", ".jpg")] = min_dis
          
        polardis_list = sorted(self.polardis.items(), key = lambda x:x[1], reverse = True)
        
        large_dis = []
        
        #print(self.sample)
        count = 0
        for itm, dis in polardis_list:
            if count < amount:
                large_dis.append(itm)
                count = count + 1  
            
        self.sample = large_dis
        #print(self.sample)
       
        
        
    
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
            
     
def run_hkmeans(x):
    """
    This function is a hierarchical 
    k-means: the centroids of current hierarchy is used
    to perform k-means in next step
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[], 'cluster2cluster':[], 'logits':[]}
    
    for seed, num_cluster in enumerate([3000,2000,1000]): #args.num_cluster
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        print(k)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 30
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 150
        clus.min_points_per_centroid = 2

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 7 #args.local_rank  
        index = faiss.GpuIndexFlatL2(res, d, cfg)  
        if seed==0: # the first hierarchy from instance directly
            clus.train(x, index)   
            D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        else:
            # the input of higher hierarchy is the centorid of lower one
            clus.train(results['centroids'][seed - 1].cpu().numpy(), index)
            D, I = index.search(results['centroids'][seed - 1].cpu().numpy(), 1)
        
        im2cluster = [int(n[0]) for n in I]
        # sample-to-centroid distances for each cluster 
        ## centroid in lower level to higher level
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            print(len(im2cluster), len(Dcluster))
            Dcluster[i].append(D[im][0])

       # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)

        if seed>0: # the im2cluster of higher hierarchy is the index of previous hierachy
            im2cluster = np.array(im2cluster) # enable batch indexing
            results['cluster2cluster'].append(torch.LongTensor(im2cluster).to(device))
            im2cluster = im2cluster[results['im2cluster'][seed - 1].cpu().numpy()]
            im2cluster = list(im2cluster)
    
        if len(set(im2cluster))==1:
            print("Warning! All samples are assigned to one cluster")

        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) 
        density = 0.2*density/density.mean() #args.T
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(device)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    
        if seed > 0: #maintain a logits from lower prototypes to higher
            proto_logits = torch.mm(results['centroids'][-1], centroids.t())
            results['logits'].append(proto_logits.to(device))


        density = torch.Tensor(density).to(device)
        im2cluster = torch.LongTensor(im2cluster).to(device)    
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

        
        
        
        

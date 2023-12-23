'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import yaml
import os
import argparse
from tqdm import tqdm

from datasets.update import g_args, update
from models import *
from models.resnet import ResNet18_cifar10
from models.vit import ViT, ViT_10
from models.swin import SwinTransformer
from models.pvt import pvtv2_b0
from utils import progress_bar, init_params
from datasets.dataset import cifar10_train, cifar10_test, cifar100_train, cifar100_test, mnist_train, mnist_test, caltech101_train, caltech101_test, caltech256_train, caltech256_test


from strategy.random_sample import random_sample
from strategy.siamese_sample import siamese_sample
from strategy.siamese_sample_hk import siamese_sample_hk
from strategy.least_confidence_sample import least_confidence_sample
from strategy.max_entropy_sample import max_entropy_sample
from strategy.margin_sample import margin_sample
from strategy.bayesian_sample import bayesian_sample
from strategy.kmean_sample import k_sample
from strategy.coreset_sample import coreset_sample

ff = open("/home/lijianing/AL/pytorch-cifar/config/cfg.yaml")
cfg = yaml.safe_load(ff)




parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')  #cifar100 0.05 cifar10 0.1 101 0.1 256 0.1
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--multi_gpu', default =False)
args = parser.parse_args()

device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')

'''
#################CIFAR data preparation########################
#set = mnist_train(cfg = cfg, mode = 'label', transform = None)
#set = caltech256_train(cfg = cfg, mode = 'label', transform = None)
#set = cifar100_train(cfg = cfg, mode = 'label', transform = transform_train_cifar100)
set = cifar10_train(cfg = cfg, mode = 'label', transform = transform_train_cifar10)
#set = caltech101_train(cfg = cfg, mode = 'label', transform = transform_train_cifar10)

'''

###############C101 data preparation#############################
#set = caltech101_train(cfg = cfg, mode = 'label', transform = transform_train_caltech101)

# Model
print('==> Building model..')
#net = VGG_cifar100('VGG16')
#net = VGG_cifar10('VGG16')
#net = VGG_caltech101('VGG16')
#net = VGG_caltech256('VGG16')
#net = resnet18_cifar10()
#net = VGG('VGG16')
net = ResNet18_cifar10()
'''
net = ViT_10(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 512,
    depth = 10,
    heads = 12,
    mlp_dim = 512,
    dropout = 0.0,
    emb_dropout = 0.0,
)
'''
'''
net = SwinTransformer(
        patch_size=2,
        hidden_dim=96,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        channels=3,
        num_classes=10,
        head_dim=32,
        window_size=7,
        downscaling_factors=(2, 2, 2, 2),
        relative_pos_embedding=True
    )

net = pvtv2_b0()
'''
net = net.to(device) 

if device == 'cuda' and args.multi_gpu == True:   #sample False while train True
    net = torch.nn.DataParallel(net, device_ids = [0,5,6,7])
    cudnn.benchmark = False



def get_sampler(strategyname, setname):
    
    set, _, _ = get_set(setname)
    if strategyname == 'random':
        
        return random_sample(set, net, cfg, dataset = setname, transform = None)
    elif strategyname == 'least_confidence':
        
        return least_confidence_sample(set, net, cfg, dataset = setname, transform = None)
    elif strategyname == 'max_entropy':
    
        return max_entropy_sample(set, net, cfg, dataset = setname, transform = None)
    elif strategyname == 'bayesian':
    
        return bayesian_sample(set, net, cfg, dataset = setname, transform = None)
    elif strategyname == 'siamese':
    
        return siamese_sample(set, net, cfg, dataset = setname, transform = None)
    
    elif strategyname == 'siamese_hk':
        
        return siamese_sample_hk(set, net, cfg, dataset = setname, transform = None)
        
    
    

'''
#######################active learning query######################
#sampler = siamese_sample(set, net, cfg, dataset = "cifar10", transform = transform_train_cifar10)
#sampler = least_confidence_sample(set, net, cfg, dataset = "caltech256", transforms = transform_train_caltech101)
#sampler = max_entropy_sample(set, net, cfg, dataset = "caltech101", transforms = transform_train_caltech101)
sampler = random_sample(set, net, cfg, dataset = "cifar10", transforms = None)
#sampler = bayesian_sample(set, net, cfg, dataset = "caltech256", transforms = transform_train_cifar100)
#sampler = k_sample(set, net, cfg, dataset = "caltech101", transforms = transform_train_cifar10)
    
sampler.query(2500)
sampler.update()
#open(cfg["dataset"]["cifar10"]["unlabel_path"], 'w').close()
update(g_args)
'''


def get_set(setname):
    set4 = caltech256_train(cfg = cfg, mode = 'label', transform = None)
    set2 = cifar100_train(cfg = cfg, mode = 'label', transform = None)
    set1 = cifar10_train(cfg = cfg, mode = 'label', transform = None)
    set3 = caltech101_train(cfg = cfg, mode = 'label', transform = None)  
    
    trainset4 = caltech256_train(cfg, mode = 'label', transform = None)
    trainset2 = cifar100_train(cfg, mode = 'label', transform = None)
    trainset3 = caltech101_train(cfg, mode = 'label', transform = None)
    trainset1 = cifar10_train(cfg, mode = 'label', transform = None)
    
    testset4 = caltech256_test(cfg, transform = None)
    testset2 = cifar100_test(cfg, transform = None)
    testset3 = caltech101_test(cfg, transform = None)
    testset1 = cifar10_test(cfg, transform = None)

    if setname == 'cifar10':
        return set1, trainset1, testset1
    elif setname == 'cifar100':
        return set2 , trainset2, testset2
    elif setname == 'caltech101':
        return set3 , trainset3, testset3
    elif setname == 'caltech256':
        return set4 , trainset4, testset4

##'#####################################################################
'''
#trainset = caltech256_train(cfg, mode = 'label', transform = None)
#trainset = mnist_train(cfg, mode = 'label', transform = None)
#trainset = cifar100_train(cfg, mode = 'label', transform = None)
#trainset = caltech101_train(cfg, mode = 'label', transform = None)
trainset = cifar10_train(cfg, mode = 'label', transform = None)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)#36 caltech  #128 cifar 256 256

#testset = mnist_test(cfg, transform = None)
#testset = caltech256_test(cfg, transform = None)
#testset = cifar100_test(cfg, transform = None)
#testset = caltech101_test(cfg, transform = None)
testset = cifar10_test(cfg, transform = None)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=200, shuffle=True, num_workers=2)

'''


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/10ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=3e-4)
                      
optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
                      
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)



# Training  cifar10 25 cifar100 40 caltech256 40
def train(epoch):
    
    if epoch > 120:
        lr = 0.01  #CIFAR0.02 256 0.02
        #lr = 0.1
        optimizer_ = optim.SGD(net.parameters(), lr=lr,
                      momentum=0.9, weight_decay=1e-3) ###0.97cifar 0.96 101
        optimizer_ = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    else:       
        optimizer_ = optimizer
    
    #optimizer_ = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)                      
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))



    # Save checkpoint.
    acc = 100.*correct/total
    f0 = open('/home/lijianing/AL/pytorch-cifar/logs/logs.txt', 'a')
    f0.write(str(acc))
    f0.write('  ')
    if epoch // 15 == 0:
        f0.write('\n')
        
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/10ckpt.pth') ####################33
        best_acc = acc


#def one_iter():




if __name__ == "__main__":    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
   
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    #optimizer = optim.SGD(net.parameters(), lr=args.lr)
      
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    '''
    sampler = get_sampler("random", "cifar10")
    sampler.query(5000)
    sampler.update()
    update(g_args)    
    '''
    set, trainset, testset = get_set("cifar10")
    trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)#36 caltech  #128 cifar 256 256

    testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=True, num_workers=2)
    
    
    for name, param in net.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param, mean=0, std=0.01)
      
    for epoch in tqdm(range(0,200)):
        
        train(epoch)
        test(epoch)
        scheduler.step()
    
    for itr in range(1,4):    
        
        sampler = get_sampler("siamese_hk", "cifar10")
        #sampler = get_sampler("max_entropy", "cifar10")
        #sampler = get_sampler("random", "cifar10")
        #sampler = get_sampler("bayesian", "cifar10")
        #sampler.query(2500, 5)
        sampler.query(2500)
        sampler.update()
        update(g_args)
        
        
        set, trainset, testset = get_set("cifar10")
        trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)#36 caltech  #128 cifar 256 256
    
        testloader = torch.utils.data.DataLoader(
        testset, batch_size=1000, shuffle=True, num_workers=2)

        for name, param in net.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.01)
                #print(name, param.data)
        

        #init_params(net)
        for epoch in tqdm(range(0,200)):
            
            train(epoch)
            test(epoch)
            scheduler.step()
  
    
                
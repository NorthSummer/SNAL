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
from utils import progress_bar, init_params
from datasets.dataset import cifar10_train, cifar10_test, cifar100_train, cifar100_test, mnist_train, mnist_test, caltech101_train, caltech101_test, caltech256_train, caltech256_test


from strategy.random_sample import random_sample
from strategy.siamese_sample import siamese_sample
from strategy.least_confidence_sample import least_confidence_sample
from strategy.max_entropy_sample import max_entropy_sample
from strategy.margin_sample import margin_sample
from strategy.bayesian_sample import bayesian_sample
from strategy.kmean_sample import k_sample
from strategy.coreset_sample import coreset_sample

ff = open("/home/lijianing/AL/pytorch-cifar/config/cfg.yaml")
cfg = yaml.safe_load(ff)




parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')  #cifar100 0.05 cifar10 0.1 101 0.1 256 0.1
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--multi_gpu', default =False)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')


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
net = ResNet18_caltech101()


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
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)



# Training  cifar10 25 cifar100 40 caltech256 40
def train(epoch):
    
    if epoch > 120:
        lr = 0.01  #CIFAR0.02 256 0.02
        #lr = 0.1
        optimizer_ = optim.SGD(net.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4) ###0.97cifar 0.96 101
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
    f0 = open('/home/lijianing/AL/pytorch-cifar/logs/logs101.txt', 'a')
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
        torch.save(state, './checkpoint/res101ckpt.pth') ####################33
        best_acc = acc


#def one_iter():
from strategy.learningloss import LossNet
loss_net = LossNet()

def train_LL(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    
    models = {"backbone": net, "module": loss_net}
   
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



if __name__ == "__main__":    


    parser = argparse.ArgumentParser('update label infomation')
    parser.add_argument('--dataset', default = 'caltech101', help = 'dataset_path_name')
    parser.add_argument('--all_path', default = '/home/lijianing/AL/pytorch-cifar/data/*/A.txt', help = 'all data dir')
    parser.add_argument('--label_path', default = '/home/lijianing/AL/pytorch-cifar/data/*/L.txt', help = 'label data dir and labels')
    parser.add_argument('--unlabel_path', default = '/home/lijianing/AL/pytorch-cifar/data/*/U.txt', help = 'unlabel data dir')
    parser.add_argument('--test_path', default = '/home/lijianing/AL/pytorch-cifar/data/*/test.txt', help = 'test data dir')
    parser.add_argument('--weight_dir', default = '/home/lijianing/AL/pytorch-cifar/*/101ckpt.pth', help = 'pretrained network parameters')
    parser.add_argument('--tensor_dir', default = '/home/lijianing/AL/pytorch-cifar/embeddings/cifar/', help = 'tensor store dir')
    t_args = parser.parse_args()
    
    
    
    t_args.all_path = t_args.all_path.replace("*", t_args.dataset)
    t_args.label_path = t_args.label_path.replace("*", t_args.dataset)
    t_args.unlabel_path = t_args.unlabel_path.replace("*", t_args.dataset)
    t_args.test_path = t_args.test_path.replace("*", t_args.dataset)
    t_args.weight_dir = t_args.weight_dir.replace("*", t_args.dataset)
    
    g_args = t_args


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    '''
    sampler = get_sampler("random", "caltech101")
    sampler.query(500)
    sampler.update()
    update(g_args)
    '''

    set, trainset, testset = get_set("caltech101")
    trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=36, shuffle=True, num_workers=2)#36 caltech  #128 cifar 256 256

    testloader = torch.utils.data.DataLoader(
    testset, batch_size=36, shuffle=True, num_workers=2)
    
    
    for epoch in tqdm(range(0,200)): 
        
        train(epoch)
        test(epoch)
        scheduler.step()
        
    for itr in range(1,6):    
        
        sampler = get_sampler("max_entropy", "caltech101")
        sampler.query(250)
        sampler.update()
        update(g_args)
      
        set, trainset, testset = get_set("caltech101")
        trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=36, shuffle=True, num_workers=2)#36 caltech  #128 cifar 256 256
    
        testloader = torch.utils.data.DataLoader(
        testset, batch_size=36, shuffle=True, num_workers=2)

        for name, param in net.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.01)

        for epoch in tqdm(range(0,200)):
        
            train(epoch)
            test(epoch)
            scheduler.step()
    
    
                
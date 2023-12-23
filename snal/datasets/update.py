import os
import argparse


parser = argparse.ArgumentParser('update label infomation')
parser.add_argument('--dataset', default = 'cifar', help = 'dataset_path_name')
parser.add_argument('--all_path', default = '/home/lijianing/AL/pytorch-cifar/data/*/A.txt', help = 'all data dir')
parser.add_argument('--label_path', default = '/home/lijianing/AL/pytorch-cifar/data/*/L.txt', help = 'label data dir and labels')
parser.add_argument('--unlabel_path', default = '/home/lijianing/AL/pytorch-cifar/data/*/U.txt', help = 'unlabel data dir')
parser.add_argument('--test_path', default = '/home/lijianing/AL/pytorch-cifar/data/*/test.txt', help = 'test data dir')
parser.add_argument('--weight_dir', default = '/home/lijianing/AL/pytorch-cifar/*/10ckpt.pth', help = 'pretrained network parameters')
parser.add_argument('--tensor_dir', default = '/home/lijianing/AL/pytorch-cifar/embeddings/cifar/', help = 'tensor store dir')
args = parser.parse_args()


global g_args
args.all_path = args.all_path.replace("*", args.dataset)
args.label_path = args.label_path.replace("*", args.dataset)
args.unlabel_path = args.unlabel_path.replace("*", args.dataset)
args.test_path = args.test_path.replace("*", args.dataset)
args.weight_dir = args.weight_dir.replace("*", args.dataset)

g_args = args

def update(args):
    f1 = open(args.all_path, 'r', encoding = 'utf-8')
    
    
    fff = open(args.unlabel_path, 'w')
    fff.close()
    
    f3 = open(args.unlabel_path, 'a')
    
    list1 = list(f1.readlines())
    
    for l1 in list1:
        l1 = l1[:-2]
    for l2 in list1:
        l2 = l2[:-2]
        
    '''
    #print(list(f2.readlines()))
    for aline in f1.readlines():
        #print(aline)
        if aline not in list(f2.readlines()):
            #print(1)
            f3.write(aline)
    '''
    
    f2 = open(args.label_path, 'r', encoding = 'utf-8')
    list2 = list(f2.readlines())
    
    for aline in list1:
        if aline not in list2:
            f3.write(aline)  




if __name__ == '__main__':
    open(args.unlabel_path, 'w').close()
    update(args)

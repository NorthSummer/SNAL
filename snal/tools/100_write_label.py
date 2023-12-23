import os
import random


def write_labels(root):
    
    train_root = os.path.join(root, 'train')
    test_root = os.path.join(root, 'test')
    
    dirs = os.listdir(test_root)
    
    f1 = open(os.path.join(root, 'test.txt'), 'a')
    
    for dir_ in dirs:
        img_root = os.path.join(test_root, dir_)
        imgs = os.listdir(img_root)
        for img in imgs:
            img_path = os.path.join(img_root, img)
            f1.write(img_path)
            f1.write(' ')
            f1.write(dir_[10:])
            f1.write('\n')
            
            
            
def shuffle(file_):     
    f1 = open(file_, 'r')
    f2 = open(file_, 'a')
    imgs = f1.readlines()
    random.shuffle(imgs)
    for img in imgs:
        f2.write(img)
        
       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    #os.remove(r'/home/liuminzhe/ljn/pytorch-cifar/data/cifar100/labels.txt')
    root = "/home/liuminzhe/ljn/pytorch-cifar/data/cifar100/"
    file_ = "/home/liuminzhe/ljn/pytorch-cifar/data/cifar100/test.txt"
    write_labels(root)
    #shuffle(file_)
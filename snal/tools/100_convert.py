import os
from skimage import io
import torchvision as tv
import numpy as np
import torch

def Cifar100(root):

    character_train = [[] for i in range(100)]
    character_test = [[] for i in range(100)]

    train_set = tv.datasets.CIFAR100(root, train=True, download=False)
    test_set = tv.datasets.CIFAR100(root, train=False, download=False)

    trainset = []
    testset = []
    for i, (X, Y) in enumerate(train_set):  
        trainset.append(list((np.array(X), np.array(Y))))
    for i, (X, Y) in enumerate(test_set):  
        testset.append(list((np.array(X), np.array(Y))))

    for X, Y in trainset:
        character_train[Y].append(X)  # 32*32*3

    for X, Y in testset:
        character_test[Y].append(X)  # 32*32*3

    os.mkdir(os.path.join(root, 'train'))
    os.mkdir(os.path.join(root, 'test'))

    for i, per_class in enumerate(character_train):
        character_path = os.path.join(root, 'train', 'character_' + str(i))
        os.mkdir(character_path)
        for j, img in enumerate(per_class):
            img_path = character_path + '/' + str(j) + ".jpg"
            io.imsave(img_path, img)

    for i, per_class in enumerate(character_test):
        character_path = os.path.join(root, 'test', 'character_' + str(i))
        os.mkdir(character_path)
        for j, img in enumerate(per_class):
            img_path = character_path + '/' + str(j) + ".jpg"
            io.imsave(img_path, img)

if __name__ == '__main__':
    root = '/home/liuminzhe/ljn/pytorch-cifar/data/cifar100'
    Cifar100(root)


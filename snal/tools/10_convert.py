import os
import numpy as np
import cv2
import json


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
 
loc_1 = '/home/liuminzhe/ljn/pytorch-cifar/data/cifar10/train_cifar10/'
loc_2 = '/home/liuminzhe/ljn/pytorch-cifar/data/cifar10/test_cifar10/'
 

if os.path.exists(loc_1) == False:
    os.mkdir(loc_1)
if os.path.exists(loc_2) == False:
    os.mkdir(loc_2)
 
 

def cifar10_img(file_dir):
    for i in range(1,6):
        data_name = file_dir + '/'+'data_batch_'+ str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')
 
        for j in range(10000):
            img = np.reshape(data_dict[b'data'][j],(3,32,32))
            img = np.transpose(img,(1,2,0))
            
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            img_name = loc_1 + str(data_dict[b'labels'][j]) + str((i)*10000 + j) + '.jpg'
            cv2.imwrite(img_name,img)
 
        print(data_name + ' is done')
 
 
    test_data_name = file_dir + '/test_batch'
    print(test_data_name + ' is processing')
    test_dict = unpickle(test_data_name)
 
    for m in range(10000):
        img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
      
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        img_name = loc_2 + str(test_dict[b'labels'][m]) + str(10000 + m) + '.jpg'
        cv2.imwrite(img_name, img)
    print(test_data_name + ' is done')
    print('Finish transforming to image')
    
anno_loc = '/home/liuminzhe/ljn/siamese_al/cifar100/annotations/'
 

if os.path.exists(anno_loc) == False:
    os.mkdir(anno_loc)
 

train_filenames = []
train_annotations = []
 
test_filenames = []
test_annotations= []
 

def cifar10_annotations(file_dir):
    print('creat train_img annotations')
    for i in range(1,6):
        data_name = file_dir + '/' + 'data_batch_' + str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')
        for j in range(10000):
            img_name = str(data_dict[b'labels'][j]) + str((i) * 10000 + j) + '.jpg'
            img_annotations = data_dict[b'labels'][j]
            train_filenames.append(img_name)
            train_annotations.append(img_annotations)
        print(data_name + ' is done')
 
    test_data_name = file_dir + '/test_batch'
    print(test_data_name + ' is processing')
    test_dict = unpickle(test_data_name)
 
    for m in range(10000):
        testimg_name = str(test_dict[b'labels'][m]) + str(10000 + m) + '.jpg'
        testimg_annotations = test_dict[b'labels'][m]     #str(test_dict[b'labels'][m])    test_dict[b'labels'][m]
        test_filenames.append(testimg_name)
        test_annotations.append(testimg_annotations)
 
    print(test_data_name + ' is done')
    print('Finish file processing')


if __name__ == '__main__':
    file_dir = '/home/liuminzhe/ljn/pytorch-cifar/data/cifar10/cifar-10-batches-py'
    cifar10_img(file_dir)
    
    file_dir = '/home/liuminzhe/ljn/pytorch-cifar/data/cifar10/cifar-10-batches-py'
    cifar10_annotations(file_dir)
 
    train_annot_dict = {
        'images': train_filenames,
        'categories': train_annotations
    }
    test_annot_dict = {
        'images':test_filenames,
        'categories':test_annotations
    }
    # print(annotation)
 
    train_json = json.dumps(train_annot_dict)
    train_file = open('/home/liuminzhe/ljn/siamese_al/cifar100/annotations/cifar10_train.json', 'w')
    train_file.write(train_json)
    train_file.close()
 
    test_json =json.dumps(test_annot_dict)
    test_file = open('/home/liuminzhe/ljn/siamese_al/cifar100/annotations/cifar10_test.json','w')
    test_file.write(test_json)
    test_file.close()
    print('annotations have writen to json file')


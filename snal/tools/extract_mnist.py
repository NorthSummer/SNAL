from PIL import Image
import struct
import time
import os
path_num_pic_in='/home/liuminzhe/ljn/pytorch-cifar/data/mnist/test_images/'
for num_dir in range(10):
    if os.path.exists(path_num_pic_in+str(num_dir)):
        print('already exists')
    else:
        os.makedirs(path_num_pic_in+str(num_dir))    


def extract_save_mnist(filename_pics,filename_labels):
    index = 0
    index2 = 0
    with open(filename_pics, 'rb') as f:
        buf=f.read()
    with open(filename_labels, 'rb') as f2:
        buf2=f2.read()
    magic, labels = struct.unpack_from('>II' , buf , index)  
    index2 += struct.calcsize('>II')   
    magic, images, rows, columns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')


    for i in range(labels):
        image = Image.new('L', (columns, rows))
        for x in range(rows):  
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')  
        label_num= struct.unpack_from('>B', buf2, index2)[0]  
        index2 += struct.calcsize('>B')  
        image.save(path_num_pic_in+str(label_num)+'/'+str(label_num)+'_'+str(time.time()) + '.jpg')
if __name__ == '__main__':
    extract_save_mnist('/home/liuminzhe/ljn/pytorch-cifar/data/mnist/MNIST/raw/t10k-images-idx3-ubyte',
                       '/home/liuminzhe/ljn/pytorch-cifar/data/mnist/MNIST/raw/t10k-labels-idx1-ubyte')
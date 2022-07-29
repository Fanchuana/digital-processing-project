import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)
#os.path.exists(file)检测file路径是否有文件或者目录，如果没有，就在file路径创建一个空文件夹

def testjpg(image):
    if (image[-3:] == 'jpg'):
        s_image = image[:-4]
    else:
        s_image = image[:-5]
    return s_image
file_path = 'mydata'
list1 = ['images', 'labels']
for cla in list1:
    mkfile('./mydata/'+ cla + '/train')
    mkfile('./mydata/'+ cla + '/val')

split_rate = 0.2 #划分比例
path1 = file_path + '/Yoloimages/'
path2 = file_path + '/Yololabels/'
images = os.listdir(path1)
num = len(images)
eval_index = random.sample(images, k=int(num * split_rate)) #随机截取images中k个元素组成新列表
for index, image in enumerate(images):#index是默认变量，代表自增索引
        if image in eval_index:
            image_path = path1 + image
            s_image = testjpg(image)
            text_path = path2 + s_image + '.txt'
            new_path1 = 'mydata/images/val/'
            new_path2 = 'mydata/labels/val/'
            copy(image_path, new_path1)
            copy(text_path, new_path2) #从老路径copy到新路径
        else:
            image_path = path1 + image
            s_image = testjpg(image)
            text_path = path2 + s_image + '.txt'
            new_path1 = 'mydata/images/train/'
            new_path2 = 'mydata/labels/train/'
            copy(image_path, new_path1)
            copy(text_path, new_path2)
print('Processing files over')

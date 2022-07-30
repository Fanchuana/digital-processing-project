import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore
import sys
from PyQt5.QtCore import *
import time

import utils
from utils import *
from models import *
import detect
import utils.general
from pathlib import Path

def init(self):
    self.unit7_img = np.ndarray(())
    self.unit7_img_channel = 1
    self.unit7_result = np.ndarray(())
    self.unit7_result_channel = 1
    self.unit7_filepath = ''
    self.unit7_imgpath = ''
    self.unit7_savepath = ''
    self.unit7_suffix = ''
    self.ui.textBrowser_6.setText('')
    self.ui.textBrowser_7.setText('')


def img_load(self):
    fileName, tmp = QFileDialog.getOpenFileName(self, '打开图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
    if fileName == '':
        return
    self.unit7_img = np.ndarray(())
    self.unit7_img_channel = 1
    self.unit7_result = np.ndarray(())
    self.unit7_result_channel = 1
    self.unit7_img = cv2.imread(fileName, -1)
    self.unit7_suffix = fileName.split('/')[-1]
    print(self.unit7_suffix)
    self.unit7_imgpath = fileName
    if self.unit7_img.size <= 1:
        return
    if len(self.unit7_img.shape) == 3:
        self.unit7_img_channel = 3
        if self.unit7_img.shape[2] == 4:
            self.unit7_img = cv2.cvtColor(self.unit7_img, cv2.COLOR_BGRA2BGR)
    print(self.unit7_img.shape)
    img_refresh(self)

def img_refresh(self):
    array = \
        [self.unit7_img,
         self.unit7_result]

    array2 = [self.ui.label_54,
              self.ui.label_55]

    channel = [self.unit7_img_channel,
               self.unit7_result_channel]
    height = 480
    weight = 500
    for index in range(len(array)):
        M = np.float32([[1, 0, 0], [0, 1, 0]])
        if array[index].size <= 1:
            array2[index].setPixmap(QtGui.QPixmap(''))
            continue
        print(array[index].shape)
        index_h = array[index].shape[0]
        index_w = array[index].shape[1]
        if index_h / index_w == height / weight:
            img = array[index].tobytes()
            if channel[index] == 1:
                image = QtGui.QImage(img, index_w, index_h, index_w * channel[index], QtGui.QImage.Format_Grayscale8)
                pix = QtGui.QPixmap.fromImage(image)
                scale_pix = pix.scaled(weight, height)
                array2[index].setPixmap(scale_pix)
                continue
            elif channel[index] == 3:
                image = QtGui.QImage(img, index_w, index_h, index_w * channel[index], QtGui.QImage.Format_BGR888)
                pix = QtGui.QPixmap.fromImage(image)
                scale_pix = pix.scaled(weight, height)
                array2[index].setPixmap(scale_pix)
                continue
        elif index_h / index_w > height / weight:
            h_ = index_h
            w_ = int(index_h * weight / height + 0.5)
            M[0, 2] += (w_ - index_w) / 2
            M[1, 2] += (h_ - index_h) / 2
        else:
            h_ = int(index_w * height / weight + 0.5)
            w_ = index_w
            M[0, 2] += (w_ - index_w) / 2
            M[1, 2] += (h_ - index_h) / 2
        img = cv2.warpAffine(array[index], M, (w_, h_))
        data = img.tobytes()
        if channel[index] == 1:
            image = QtGui.QImage(data, w_, h_, w_ * channel[index], QtGui.QImage.Format_Grayscale8)
            pix = QtGui.QPixmap.fromImage(image)
            scale_pix = pix.scaled(weight, height)
            array2[index].setPixmap(scale_pix)
            continue
        else:
            image = QtGui.QImage(data, w_, h_, w_ * channel[index], QtGui.QImage.Format_BGR888)
            pix = QtGui.QPixmap.fromImage(image)
            scale_pix = pix.scaled(weight, height)
            array2[index].setPixmap(scale_pix)
            continue
    return


def result_save(self):
    fileName= QFileDialog.getExistingDirectory(self, '保存图像')
    if fileName == '':
            return
    self.unit7_savepath = fileName
    self.ui.textBrowser_7.setText(fileName.split('/')[-2]+'/'+fileName.split('/')[-1])
    msg_box = QMessageBox(QMessageBox.Information, '成功', '选择路径成功,保存路径为：' + fileName)
    msg_box.exec_()

def clear(self):
    if self.unit7_img.size > 1:
        init(self)
        img_refresh(self)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '无需清空', '没有图片')
        msg_box.exec_()


def result_show(self):
    if self.unit7_result.size > 1:
        cv2.imshow('Original pic', self.unit7_result)
        cv2.waitKey(0)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '没有生成图像')
        msg_box.exec_()


def object_detection(self):
    if self.unit7_filepath !='' and self.unit7_img.size>1 and self.unit7_savepath!='':
        modelpath = self.unit7_filepath
        imgpath = self.unit7_imgpath
        savepath = self.unit7_savepath
        detect.main(imgpath, modelpath,savepath)
        name ='exp'
        z = utils.general.increment_path_num(Path(savepath) / name, exist_ok=False)
        num = str(z) if z!=1 else ''
        path = savepath +'/exp'+ num+'/'+self.unit7_suffix
        print(path)
        self.unit7_result = cv2.imread(path, -1)
        if self.unit7_result.size >1:
           if len(self.unit7_result.shape) == 3:
               self.unit7_result_channel = 3
               if self.unit7_result.shape[2] == 4:
                   self.unit7_result = cv2.cvtColor(self.unit7_result, cv2.COLOR_BGRA2BGR)
           print(self.unit7_result.shape)
           img_refresh(self)
        else:
            msg_box = QMessageBox(QMessageBox.Warning, 'error', 'error1')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有导入模型或图片', '请导入模型和图片后再进行尝试')
        msg_box.exec_()




def model_load(self):
    fileName, tmp = QFileDialog.getOpenFileName(self, '选择模型路径', 'Model', '*.pt')
    if fileName == '':
        return
    self.unit7_filepath = fileName
    self.ui.textBrowser_6.setText(fileName.split('/')[-2]+'/'+fileName.split('/')[-1])
    print(self.unit7_filepath)
    if self.unit7_filepath =='':
        return
    else:
        msg_box = QMessageBox(QMessageBox.Information, '已检测到模型', '模型导入成功')
        msg_box.exec_()



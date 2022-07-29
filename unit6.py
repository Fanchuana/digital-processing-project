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

import util
from util import *
from model import *

#images = np.ndarray(())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#class Runthread(QtCore.QThread):
    #  通过类成员对象定义信号对象
   # signal = pyqtSignal(int)
  #  filename2 = ''
  #  def __init__(self):
     #   super(Runthread, self).__init__()

  #  def __del__(self):
     #   self.wait()

  #  def setParam(self, file1, file2):
       # self.filename1 = file1
       # self.filename2 = file2




def init(self):
    self.unit6_img1 = np.ndarray(())
    self.unit6_img2 = np.ndarray(())
    self.unit6_result = np.ndarray(())
    self.unit6_img1_channel = 3
    self.unit6_img2_channel = 3
    self.unit6_result_channel = 3
    self.filepath1 =''
    self.filepath2 =''
    self.ui.progressBar.setValue(0)
    self.ui.progressBar.setMaximum(100)

def img_load1(self):
    fileName, tmp = QFileDialog.getOpenFileName(self, '打开图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
    if fileName == '':
        return
    self.filepath1 = fileName
    self.unit6_img1 = cv2.imread(fileName, -1)
    if len(self.unit6_img1.shape) == 3:
        self.unit6_img1_channel = 3
        if self.unit6_img1.shape[2] == 4:
            self.unit6_img1 = cv2.cvtColor(self.unit6_img1, cv2.COLOR_BGRA2BGR)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, "不是彩图", '请选择彩图进行风格迁移  ')
        msg_box.exec_()
        init(self)
        return
    print(self.unit6_img1.shape)
    img_refresh(self)

def img_load2(self):
    fileName, tmp = QFileDialog.getOpenFileName(self, '打开图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
    if fileName == '':
        return
    self.filepath2 = fileName
    self.unit6_img2 = cv2.imread(fileName, -1)
    if len(self.unit6_img2.shape) != 3:
        msg_box = QMessageBox(QMessageBox.Warning, "不是彩图", '请选择彩图进行风格迁移  ')
        msg_box.exec_()
        init(self)
        return
    else:
        self.unit6_img2_channel = 3
        if self.unit6_img2.shape[2] == 4:
            self.unit6_img2 = cv2.cvtColor(self.unit6_img2, cv2.COLOR_BGRA2BGR)
    print(self.unit6_img1.shape)
    img_refresh(self)

def img_refresh(self):
    array = \
        [self.unit6_img1,
         self.unit6_img2,
         self.unit6_result]

    array2 = [self.ui.label_44,
              self.ui.label_45,
              self.ui.label_47
            ]

    channel = [self.unit6_img1_channel,
               self.unit6_img2_channel,
               self.unit6_result_channel]
    height = 350
    weight = 350
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

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def style_transfer(self):
    if self.unit6_img1.size>1 or self.unit6_img2.size>1:
        style_img = read_image(self.filepath2, target_width=512).to(device)
        print(torch.cuda.is_available())
        content_img = read_image(self.filepath1, target_width=512).to(device)
        print(style_img.shape)
        print(content_img.shape)
        vgg16 = models.vgg16(pretrained=True)
        vgg16 = VGG(vgg16.features[:23]).to(device).eval()
        style_features = vgg16(style_img)
        content_features = vgg16(content_img)
        style_grams = [gram_matrix(x) for x in style_features]
        input_img = content_img.clone()
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        print("Yes2")
        style_weight = 1e6
        content_weight = 1
        run = [0]
        print("Yes3")
        while run[0] <= 300:
            QApplication.processEvents()
            self.ui.progressBar.setValue(int(run[0]/3))
            # if(run[0]%3==0):
            # self.signal.emit(int(run[0]/3))
            def f():
                optimizer.zero_grad()
                features = vgg16(input_img)
                content_loss = F.mse_loss(features[2], content_features[2]) * content_weight
                style_loss = 0
                grams = [gram_matrix(x) for x in features]
                for a, b in zip(grams, style_grams):
                    style_loss += F.mse_loss(a, b) * style_weight
                loss = style_loss + content_loss
                if run[0] % 50 == 0:
                    print('Step {}: Style Loss: {:4f} Content Loss: {:4f}'.format(
                        run[0], style_loss.item(), content_loss.item()))
                run[0] += 1
                loss.backward()
                return loss

            optimizer.step(f)
        self.unit6_result = util.recover_image(input_img)
        img_refresh(self)
        print("Train over!")
      #  thread = Runthread()
      #  thread.setParam(self.filepath1, self.filepath2)
       # print("Yes0")
       # try:
      #   thread.signal.connect(self.progressBar_refresh)
      #  except:
       #   print("Yes1")
      #  thread.start()
      #  print("Yes2")
      #  thread.wait()
     #   print("Yes3")
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '无需清空', '没有图片')
        msg_box.exec_()


#def progressBar_refresh(self, msg):
  #  self.ui.progressBar.setValue(int(msg))



def img_save(self):
    if self.unit6_result.size > 1:
        fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        if fileName == '':
            return
        cv2.imwrite(fileName, self.unit6_result)
        msg_box = QMessageBox(QMessageBox.Information, '成功', '图像保存成功,保存路径为：' + fileName)
        msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '没有生成图像')
        msg_box.exec_()



def img_clear(self):
    if self.unit6_img1.size > 1 or self.unit6_img2.size > 1:
        init(self)
        img_refresh(self)

    else:
        msg_box = QMessageBox(QMessageBox.Warning, '无需清空', '没有图片')
        msg_box.exec_()


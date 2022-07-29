from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import cv2
import numpy as np

def init(self):
    self.unit3_img1 = np.ndarray(())
    self.unit3_img2 = np.ndarray(())
    self.unit3_result1 = np.ndarray(())
    self.unit3_img1_channel = 1
    self.unit3_img2_channel = 1
    self.unit3_result1_channel = 1
    self.unit3_img3 = np.ndarray(())
    self.unit3_result2 = np.ndarray(())
    self.unit3_img3_channel = 1
    self.unit3_result2_channel = 1
def unit3_img_refresh(self):
        array = [self.unit3_img1, self.unit3_img2, self.unit3_result1, self.unit3_img3, self.unit3_result2]
        array2 = [self.ui.label_35, self.ui.label_36, self.ui.label_29, self.ui.label_33, self.ui.label_31]
        channel = [self.unit3_img1_channel,
                   self.unit3_img2_channel,
                   self.unit3_result1_channel,
                   self.unit3_img3_channel, self.unit3_result2_channel]
        for index in range(len(array)):
            M = np.float32([[1, 0, 0], [0, 1, 0]])
            if array[index].size <=1 :
                continue
            print(array[index].shape)
            index_h = array[index].shape[0]
            index_w = array[index].shape[1]
            if index_h / index_w == 250 / 200:
               img = array[index].tobytes()
               if channel[index] == 1:
                 image = QtGui.QImage(img, index_w, index_h, index_w * channel[index], QtGui.QImage.Format_Grayscale8)
                 pix = QtGui.QPixmap.fromImage(image)
                 scale_pix = pix.scaled(200, 250)
                 array2[index].setPixmap(scale_pix)
                 continue
               elif channel[index] == 3:
                 image = QtGui.QImage(img, index_w, index_h, index_w * channel[index], QtGui.QImage.Format_BGR888)
                 pix = QtGui.QPixmap.fromImage(image)
                 scale_pix = pix.scaled(200, 250)
                 array2[index].setPixmap(scale_pix)
                 continue
            elif index_h / index_w > 250 / 200:
               h_ = index_h
               w_ = int(index_h * 200 / 250 + 0.5)
               M[0, 2] += (w_ - index_w) / 2
               M[1, 2] += (h_ - index_h) / 2
            else:
               h_ = int(index_w * 250 / 200 + 0.5)
               w_ = index_w
               M[0, 2] += (w_ - index_w) / 2
               M[1, 2] += (h_ - index_h) / 2
            img = cv2.warpAffine(array[index], M, (w_, h_))
            data = img.tobytes()
            if channel[index] == 1:
                image = QtGui.QImage(data, w_, h_, w_* channel[index], QtGui.QImage.Format_Grayscale8)
                pix = QtGui.QPixmap.fromImage(image)
                scale_pix = pix.scaled(200, 250)
                array2[index].setPixmap(scale_pix)
                continue
            else:
                image = QtGui.QImage(data, w_, h_, w_ * channel[index], QtGui.QImage.Format_BGR888)
                pix = QtGui.QPixmap.fromImage(image)
                scale_pix = pix.scaled(200, 250)
                array2[index].setPixmap(scale_pix)
                continue
        return

def img_left_load1(self):
      fileName, tmp = QFileDialog.getOpenFileName(self, '打开图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
      if fileName == '':
          return
      self.unit3_img1 = cv2.imread(fileName, -1)
      if self.unit3_img1.size <= 1:
          return
      if len(self.unit3_img1.shape)==3:
          self.unit3_img1_channel =3
          if self.unit3_img1.shape[2]==4:
              self.unit3_img1 = cv2.cvtColor(self.unit3_img1, cv2.COLOR_BGRA2BGR)
      print(self.unit3_img1.shape)
      unit3_img_refresh(self)


def img_left_load2(self):
    fileName, tmp = QFileDialog.getOpenFileName(self, '打开图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
    if fileName == '':
        return
    self.unit3_img2 = cv2.imread(fileName, -1)
    if self.unit3_img2.size <= 1:
        return
    if len(self.unit3_img2.shape) == 3:
        self.unit3_img2_channel = 3
        if self.unit3_img2.shape[2] == 4:
            self.unit3_img2 = cv2.cvtColor(self.unit3_img2, cv2.COLOR_BGRA2BGR)
    print(self.unit3_img2.shape)
    unit3_img_refresh(self)


def ADD(self):
    if self.unit3_img1.size>1 and self.unit3_img2.size>1 \
            and self.unit3_img1_channel == self.unit3_img2_channel\
            and self.unit3_img1.shape == self.unit3_img2.shape:
        try:
            self.unit3_result1 = cv2.add(src1=self.unit3_img1, src2=self.unit3_img2)
        except:
            msg_box = QMessageBox(QMessageBox.Warning, '图片异常', '请重新选择图片进行加操作')
            msg_box.exec_()
        if len(self.unit3_result1.shape) == 3:
            self.unit3_result1.channel = 3
            if self.unit3_result1.shape[2] == 4:
                self.unit3_result1 = cv2.cvtColor(self.unit3_result1, cv2.COLOR_BGRA2BGR)
    elif self.unit3_img1.size<=1 or self.unit3_img2.size<=1:
        msg_box = QMessageBox(QMessageBox.Warning, '缺失图片', '请选择两张图片后再相加')
        msg_box.exec_()
    elif self.unit3_img1_channel != self.unit3_img2_channel:
        msg_box = QMessageBox(QMessageBox.Warning, '通道不同', '彩色图与灰度图不能相加')
        msg_box.exec_()
    elif self.unit3_img1.shape != self.unit3_img2.shape:
        msg_box = QMessageBox(QMessageBox.Warning, '图片尺寸不同', '尺寸不同的图像不能相加')
        msg_box.exec_()
    unit3_img_refresh(self)

def SUB(self):
    if self.unit3_img1.size > 1 and self.unit3_img2.size > 1 \
            and self.unit3_img1_channel == self.unit3_img2_channel \
            and self.unit3_img1.shape == self.unit3_img2.shape:
        try:
            self.unit3_result1 = cv2.subtract(src1=self.unit3_img1, src2=self.unit3_img2)
        except:
            msg_box = QMessageBox(QMessageBox.Warning, '图片异常', '请重新选择图片进行减操作')
            msg_box.exec_()
        if len(self.unit3_result1.shape) == 3:
            self.unit3_result1.channel = 3
            if self.unit3_result1.shape[2] == 4:
                self.unit3_result1 = cv2.cvtColor(self.unit3_result1, cv2.COLOR_BGRA2BGR)
    elif self.unit3_img1.size <= 1 or self.unit3_img2.size <= 1:
        msg_box = QMessageBox(QMessageBox.Warning, '缺失图片', '请选择两张图片后再相减')
        msg_box.exec_()
    elif self.unit3_img1_channel != self.unit3_img2_channel:
        msg_box = QMessageBox(QMessageBox.Warning, '通道不同', '彩色图与灰度图不能相减')
        msg_box.exec_()
    elif self.unit3_img1.shape != self.unit3_img2.shape:
        msg_box = QMessageBox(QMessageBox.Warning, '图片尺寸不同', '尺寸不同的图像不能相减')
        msg_box.exec_()
    unit3_img_refresh(self)


def MULTI(self):
    if self.unit3_img1.size > 1 and self.unit3_img2.size > 1 \
            and self.unit3_img1_channel == self.unit3_img2_channel \
            and self.unit3_img1.shape == self.unit3_img2.shape:
        try:
            self.unit3_result1 = cv2.multiply(src1=self.unit3_img1, src2=self.unit3_img2)
        except:
            msg_box = QMessageBox(QMessageBox.Warning, '图片异常', '请重新选择图片进行乘操作')
            msg_box.exec_()
        if len(self.unit3_result1.shape) == 3:
            self.unit3_result1.channel = 3
            if self.unit3_result1.shape[2] == 4:
                self.unit3_result1 = cv2.cvtColor(self.unit3_result1, cv2.COLOR_BGRA2BGR)
    elif self.unit3_img1.size <= 1 or self.unit3_img2.size <= 1:
        msg_box = QMessageBox(QMessageBox.Warning, '缺失图片', '请选择两张图片后再相乘')
        msg_box.exec_()
    elif self.unit3_img1_channel != self.unit3_img2_channel:
        msg_box = QMessageBox(QMessageBox.Warning, '通道不同', '彩色图与灰度图不能相乘')
        msg_box.exec_()
    elif self.unit3_img1.shape != self.unit3_img2.shape:
        msg_box = QMessageBox(QMessageBox.Warning, '图片尺寸不同', '尺寸不同的图像不能相乘')
        msg_box.exec_()
    unit3_img_refresh(self)


def DIVIDE(self):
    if self.unit3_img1.size > 1 and self.unit3_img2.size > 1 \
            and self.unit3_img1_channel == self.unit3_img2_channel \
            and self.unit3_img1.shape == self.unit3_img2.shape:
        try:
          self.unit3_result1 = cv2.divide(src1=self.unit3_img1, src2=self.unit3_img2)
        except:
            msg_box = QMessageBox(QMessageBox.Warning, '图片异常', '图片中有灰度值为0的像素点')
            msg_box.exec_()
        if len(self.unit3_result1.shape) == 3:
            self.unit3_result1.channel = 3
            if self.unit3_result1.shape[2] == 4:
                self.unit3_result1 = cv2.cvtColor(self.unit3_result1, cv2.COLOR_BGRA2BGR)
    elif self.unit3_img1.size <= 1 or self.unit3_img2.size <= 1:
        msg_box = QMessageBox(QMessageBox.Warning, '缺失图片', '请选择两张图片后再相除')
        msg_box.exec_()
    elif self.unit3_img1_channel != self.unit3_img2_channel:
        msg_box = QMessageBox(QMessageBox.Warning, '通道不同', '彩色图与灰度图不能相除')
        msg_box.exec_()
    elif self.unit3_img1.shape != self.unit3_img2.shape:
        msg_box = QMessageBox(QMessageBox.Warning, '图片尺寸不同', '尺寸不同的图像不能相除')
        msg_box.exec_()
    unit3_img_refresh(self)


def AND(self):
    if self.unit3_img1.size > 1 and self.unit3_img2.size > 1 \
            and self.unit3_img1_channel == self.unit3_img2_channel \
            and self.unit3_img1.shape == self.unit3_img2.shape:
        try:
            self.unit3_result1 = self.unit3_img1&self.unit3_img2
        except:
            msg_box = QMessageBox(QMessageBox.Warning, '图片异常', '请重新选择图片进行与操作')
            msg_box.exec_()
        if len(self.unit3_result1.shape) == 3:
            self.unit3_result1.channel = 3
            if self.unit3_result1.shape[2] == 4:
                self.unit3_result1 = cv2.cvtColor(self.unit3_result1, cv2.COLOR_BGRA2BGR)
    elif self.unit3_img1.size <= 1 or self.unit3_img2.size <= 1:
        msg_box = QMessageBox(QMessageBox.Warning, '缺失图片', '请选择两张图片再进行与操作')
        msg_box.exec_()
    elif self.unit3_img1_channel != self.unit3_img2_channel:
        msg_box = QMessageBox(QMessageBox.Warning, '通道不同', '彩色图与灰度图不能进行与操作')
        msg_box.exec_()
    elif self.unit3_img1.shape != self.unit3_img2.shape:
        msg_box = QMessageBox(QMessageBox.Warning, '图片尺寸不同', '尺寸不同的图像不能进行与操作')
        msg_box.exec_()
    unit3_img_refresh(self)


def OR(self):
    if self.unit3_img1.size > 1 and self.unit3_img2.size > 1 \
            and self.unit3_img1_channel == self.unit3_img2_channel \
            and self.unit3_img1.shape == self.unit3_img2.shape:
        try:
            self.unit3_result1 = self.unit3_img1 | self.unit3_img2
        except:
            msg_box = QMessageBox(QMessageBox.Warning, '图片异常', '请重新选择图片进行或操作')
            msg_box.exec_()
        if len(self.unit3_result1.shape) == 3:
            self.unit3_result1.channel = 3
            if self.unit3_result1.shape[2] == 4:
                self.unit3_result1 = cv2.cvtColor(self.unit3_result1, cv2.COLOR_BGRA2BGR)
    elif self.unit3_img1.size <= 1 or self.unit3_img2.size <= 1:
        msg_box = QMessageBox(QMessageBox.Warning, '缺失图片', '请选择两张图片后再进行或操作')
        msg_box.exec_()
    elif self.unit3_img1_channel != self.unit3_img2_channel:
        msg_box = QMessageBox(QMessageBox.Warning, '通道不同', '彩色图与灰度图不能进行或操作')
        msg_box.exec_()
    elif self.unit3_img1.shape != self.unit3_img2.shape:
        msg_box = QMessageBox(QMessageBox.Warning, '图片尺寸不同', '尺寸不同的图像不能进行或操作')
        msg_box.exec_()
    unit3_img_refresh(self)


def NOT(self):
    if self.unit3_img1.size>1:
        try:
            self.unit3_result1 = ~self.unit3_img1
        except:
            msg_box = QMessageBox(QMessageBox.Warning, '图片异常', '请重新选择图片进行或操作')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '缺失图片', '请选择两张图片后再进行非操作')
        msg_box.exec_()
    unit3_img_refresh(self)





def img_left_clear(self):
    if self.unit3_img1.size > 1 or self.unit3_img2.size > 1:
        self.unit3_img1 = np.ndarray(())
        self.unit3_img2 = np.ndarray(())
        self.unit3_result1 = np.ndarray(())
        self.unit3_img1_channel = 1
        self.unit3_img2_channel = 1
        self.unit3_result1_channel = 1
        for label in [self.ui.label_35, self.ui.label_36, self.ui.label_29]:
            label.setPixmap(QtGui.QPixmap(''))
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '无需清空', '没有图片')
        msg_box.exec_()

    unit3_img_refresh(self)


def img_left_save(self):
    if self.unit3_result1.size > 1:
        fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        if fileName == '':
            return
        cv2.imwrite(fileName, self.unit3_result1)
        msg_box = QMessageBox(QMessageBox.Information, '成功', '图像保存成功,保存路径为：' + fileName)
        msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '没有生成图像')
        msg_box.exec_()


def img_left_show(self):
    if self.unit3_result1 > 1:
        cv2.imshow('Original pic', self.unit3_result1)
        cv2.waitKey(0)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '没有生成图像')
        msg_box.exec_()


def img_right_load(self):
    fileName, tmp = QFileDialog.getOpenFileName(self, '打开图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
    if fileName == '':
        return
    self.unit3_img3 = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    if self.unit3_img3.size <= 1:
        return
    print(self.unit3_img3.shape)
    unit3_img_refresh(self)


def erode(self):
    checked = self.ui.radioButton.isChecked()
    try:
      x = self.ui.lineEdit_20.text()
      y = self.ui.lineEdit_27.text()
    except:
      msg_box = QMessageBox(QMessageBox.Warning, '结构元不能为空', '请重新输入')
      msg_box.exec_()
      return
    if self.unit3_img3.size>1:
      if x and y:
          x = int(x)
          y = int(y)
          kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (x, y), (-1, -1))
          if checked:
              if self.unit3_result2.size>1:
                  self.unit3_result2 = cv2.erode(self.unit3_result2, kernel)
              else:
                  self.unit3_result2 = cv2.erode(self.unit3_img3, kernel)
          else:
              self.unit3_result2 = cv2.erode(self.unit3_img3, kernel)
          if len(self.unit3_result2.shape) == 3:
                  self.unit3_result2.channel = 3
                  if self.unit3_result2.shape[2] == 4:
                      self.unit3_result3 = cv2.cvtColor(self.unit3_result2, cv2.COLOR_BGRA2BGR)
          unit3_img_refresh(self)
      else:
          msg_box = QMessageBox(QMessageBox.Warning, '结构元为空', '请输入结构元大小')
          msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像')
        msg_box.exec_()


def dilate(self):
    checked = self.ui.radioButton.isChecked()
    try:
      x = self.ui.lineEdit_20.text()
      y = self.ui.lineEdit_27.text()
    except:
      msg_box = QMessageBox(QMessageBox.Warning, '结构元不能为空', '请重新输入')
      msg_box.exec_()
      return
    if self.unit3_img3.size > 1:
        if x and y:
            x = int(x)
            y = int(y)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (x, y), (-1, -1))
            if checked:
                if self.unit3_result2.size > 1:
                    self.unit3_result2 = cv2.dilate(self.unit3_result2, kernel)
                else:
                    self.unit3_result2 = cv2.dilate(self.unit3_img3, kernel)
            else:
                self.unit3_result2 = cv2.dilate(self.unit3_img3, kernel)
            if len(self.unit3_result2.shape) == 3:
                self.unit3_result2.channel = 3
                if self.unit3_result2.shape[2] == 4:
                    self.unit3_result3 = cv2.cvtColor(self.unit3_result2, cv2.COLOR_BGRA2BGR)
            unit3_img_refresh(self)
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '结构元为空', '请输入结构元大小')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像')
        msg_box.exec_()


def opening(self):
    checked = self.ui.radioButton.isChecked()
    try:
      x = self.ui.lineEdit_20.text()
      y = self.ui.lineEdit_27.text()
    except:
      msg_box = QMessageBox(QMessageBox.Warning, '结构元不能为空', '请重新输入')
      msg_box.exec_()
      return
    if self.unit3_img3.size > 1:
        if x and y:
            x = int(x)
            y = int(y)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (x, y), (-1, -1))
            if checked:
                if self.unit3_result2.size > 1:
                    self.unit3_result2 = cv2.morphologyEx(self.unit3_result2, cv2.MORPH_OPEN, kernel)
                else:
                    self.unit3_result2 = cv2.morphologyEx(self.unit3_img3,cv2.MORPH_OPEN, kernel)
            else:
                self.unit3_result2 = cv2.morphologyEx(self.unit3_img3, cv2.MORPH_OPEN, kernel)
            if len(self.unit3_result2.shape) == 3:
                self.unit3_result2.channel = 3
                if self.unit3_result2.shape[2] == 4:
                    self.unit3_result3 = cv2.cvtColor(self.unit3_result2, cv2.COLOR_BGRA2BGR)
            unit3_img_refresh(self)
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '结构元为空', '请输入结构元大小')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像')
        msg_box.exec_()


def closing(self):
    checked = self.ui.radioButton.isChecked()
    try:
      x = self.ui.lineEdit_20.text()
      y = self.ui.lineEdit_27.text()
    except:
      msg_box = QMessageBox(QMessageBox.Warning, '结构元不能为空', '请重新输入')
      msg_box.exec_()
      return
    if self.unit3_img3.size > 1:
        if x and y:
            x = int(x)
            y = int(y)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (x, y), (-1, -1))
            if checked:
                if self.unit3_result2.size > 1:
                    self.unit3_result2 = cv2.morphologyEx(self.unit3_result2, cv2.MORPH_CLOSE, kernel)
                else:
                    self.unit3_result2 = cv2.morphologyEx(self.unit3_img3, cv2.MORPH_CLOSE, kernel)
            else:
                self.unit3_result2 = cv2.morphologyEx(self.unit3_img3, cv2.MORPH_CLOSE, kernel)
            if len(self.unit3_result2.shape) == 3:
                self.unit3_result2.channel = 3
                if self.unit3_result2.shape[2] == 4:
                    self.unit3_result3 = cv2.cvtColor(self.unit3_result2, cv2.COLOR_BGRA2BGR)
            unit3_img_refresh(self)
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '结构元为空', '请输入结构元大小')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像')
        msg_box.exec_()


def mean(self):
    checked = self.ui.radioButton.isChecked()
    try:
      x = self.ui.lineEdit_20.text()
      y = self.ui.lineEdit_27.text()
    except:
      msg_box = QMessageBox(QMessageBox.Warning, '结构元不能为空', '请重新输入')
      msg_box.exec_()
      return
    if self.unit3_img3.size > 1:
        if x and y:
            x = int(x)
            y = int(y)
            if x<=0 or y<=0 or x!=y or x%2!=1:
                msg_box = QMessageBox(QMessageBox.Warning, '均值滤波长宽均为正奇数', '请重新输入')
                msg_box.exec_()
                return
            if checked:
                if self.unit3_result2.size > 1:
                    self.unit3_result2 = cv2.blur(self.unit3_result2, (x,y),(-1,-1))
                else:
                    self.unit3_result2 = cv2.blur(self.unit3_img3, (x,y),(-1,-1))
            else:
                self.unit3_result2 = cv2.blur(self.unit3_img3, (x,y),(-1,-1))
            if len(self.unit3_result2.shape) == 3:
                self.unit3_result2.channel = 3
                if self.unit3_result2.shape[2] == 4:
                    self.unit3_result3 = cv2.cvtColor(self.unit3_result2, cv2.COLOR_BGRA2BGR)
            unit3_img_refresh(self)
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '结构元为空', '请输入结构元大小')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像')
        msg_box.exec_()


def guassian(self):
    checked = self.ui.radioButton.isChecked()
    try:
        x = self.ui.lineEdit_20.text()
        y = self.ui.lineEdit_27.text()
    except:
        msg_box = QMessageBox(QMessageBox.Warning, '结构元不能为空', '请重新输入')
        msg_box.exec_()
        return
    if self.unit3_img3.size > 1:
        if x and y:
            x = int(x)
            y = int(y)
            if x<=0 or y<=0 or x!=y or x%2!=1:
                msg_box = QMessageBox(QMessageBox.Warning, '高斯滤波长宽均为正奇数', '请重新输入')
                msg_box.exec_()
                return
            if checked:
                if self.unit3_result2.size > 1:
                    self.unit3_result2 = cv2.GaussianBlur(self.unit3_result2, (x, y), 0,0)
                else:
                    self.unit3_result2 = cv2.GaussianBlur(self.unit3_img3, (x, y), 0,0)
            else:
                self.unit3_result2 = cv2.GaussianBlur(self.unit3_img3, (x, y), 0,0)
            if len(self.unit3_result2.shape) == 3:
                self.unit3_result2.channel = 3
                if self.unit3_result2.shape[2] == 4:
                    self.unit3_result3 = cv2.cvtColor(self.unit3_result2, cv2.COLOR_BGRA2BGR)
            unit3_img_refresh(self)
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '结构元为空', '请输入结构元大小')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像')
        msg_box.exec_()


def Covfilter(self):
    checked = self.ui.radioButton.isChecked()
    try:
      x = self.ui.lineEdit_20.text()
      y = self.ui.lineEdit_27.text()
    except:
      msg_box = QMessageBox(QMessageBox.Warning, '结构元不能为空', '请重新输入')
      msg_box.exec_()
      return
    if self.unit3_img3.size > 1:
        if x and y:
            x = int(x)
            y = int(y)
            if x <= 0 or y <= 0  or x % 2 != 1:
                msg_box = QMessageBox(QMessageBox.Warning, '卷积核长宽均为正奇数', '请重新输入')
                msg_box.exec_()
                return
            kernel = np.ones((x, y), np.float32) / (x*y)
            if checked:
                if self.unit3_result2.size > 1:
                    self.unit3_result2 = cv2.filter2D(self.unit3_result2, -1, kernel)
                else:
                    self.unit3_result2 = cv2.filter2D(self.unit3_img3,  -1, kernel)
            else:
                self.unit3_result2 = cv2.filter2D(self.unit3_img3,  -1, kernel)
            if len(self.unit3_result2.shape) == 3:
                self.unit3_result2.channel = 3
                if self.unit3_result2.shape[2] == 4:
                    self.unit3_result3 = cv2.cvtColor(self.unit3_result2, cv2.COLOR_BGRA2BGR)
            unit3_img_refresh(self)
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '结构元为空', '请输入结构元大小')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像')
        msg_box.exec_()


def median(self):
    checked = self.ui.radioButton.isChecked()
    try:
      x = self.ui.lineEdit_20.text()
      y = self.ui.lineEdit_27.text()
    except:
      msg_box = QMessageBox(QMessageBox.Warning, '结构元不能为空', '请重新输入')
      msg_box.exec_()
      return
    if self.unit3_img3.size > 1:
        if x and y:
            x = int(x)
            y = int(y)
            if x<=0 or y<=0 or x!=y or x%2!=1:
                msg_box = QMessageBox(QMessageBox.Warning, '中值滤波算子长宽均为正奇数且相等', '请重新输入')
                msg_box.exec_()
                return
            print(x,y)
            if checked:
                if self.unit3_result2.size > 1:
                    self.unit3_result2 = cv2.medianBlur(self.unit3_result2, x)
                else:
                    self.unit3_result2 = cv2.medianBlur(self.unit3_img3, x)
            else:
                self.unit3_result2 = cv2.medianBlur(self.unit3_img3, x)
            if len(self.unit3_result2.shape) == 3:
                self.unit3_result2.channel = 3
                if self.unit3_result2.shape[2] == 4:
                    self.unit3_result3 = cv2.cvtColor(self.unit3_result2, cv2.COLOR_BGRA2BGR)
            unit3_img_refresh(self)
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '结构元为空', '请输入结构元大小')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像')
        msg_box.exec_()


def img_right_clear(self):
    if self.unit3_img3.size > 1 or self.unit3_result2.size > 1:
        self.unit3_img3 = np.ndarray(())
        self.unit3_result2 = np.ndarray(())
        self.unit3_img3_channel = 1
        self.unit3_result2_channel = 1
        for label in [self.ui.label_33, self.ui.label_31]:
            label.setPixmap(QtGui.QPixmap(''))
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '无需清空', '没有图片')
        msg_box.exec_()
    unit3_img_refresh(self)

def bilateralFilter(self):
    checked = self.ui.radioButton.isChecked()
    try:
       d = self.ui.lineEdit_11.text()
       sigmaSpace = self.ui.lineEdit_12.text()
       sigmaColor = self.ui.lineEdit_13.text()
    except:
       msg_box = QMessageBox(QMessageBox.Warning, '结构元不能为空', '请重新输入')
       msg_box.exec_()
       return
    if self.unit3_img3.size > 1:
        if d and sigmaSpace and sigmaColor:
            d = int(d)
            sigmaSpace = int(sigmaSpace)
            sigmaColor = int(sigmaColor)
            if d <= 0 or sigmaColor <= 0 or sigmaSpace <= 0:
                msg_box = QMessageBox(QMessageBox.Warning, '中值滤波算子长宽均为正奇数且相等', '请重新输入')
                msg_box.exec_()
                return
            if checked:
                if self.unit3_result2.size > 1:
                    self.unit3_result2 = cv2.bilateralFilter(self.unit3_result2, d, sigmaColor, sigmaSpace)
                else:
                    self.unit3_result2 = cv2.bilateralFilter(self.unit3_img3, d, sigmaColor, sigmaSpace)
            else:
                self.unit3_result2 = cv2.bilateralFilter(self.unit3_img3, d, sigmaColor, sigmaSpace)
            if len(self.unit3_result2.shape) == 3:
                self.unit3_result2.channel = 3
                if self.unit3_result2.shape[2] == 4:
                    self.unit3_result3 = cv2.cvtColor(self.unit3_result2, cv2.COLOR_BGRA2BGR)
            unit3_img_refresh(self)
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '结构元或方差为空', '请输入结构元大小和方差')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
        msg_box.exec_()


def add_noise_Guass(self):  # 添加高斯噪声
    if self.unit3_img3.size>1:
        mu = 0.0
        sigma = 0.1
        self.unit3_result2 = np.array(self.unit3_img3 / 255, dtype=float)
        noise = np.random.normal(mu, sigma, self.unit3_result2.shape)
        self.unit3_result2 = self.unit3_result2 + noise
        if self.unit3_result2.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        self.unit3_result2 = np.clip(self.unit3_result2, low_clip, 1.0)
        self.unit3_result2 = np.uint8(self.unit3_result2 * 255)
        unit3_img_refresh(self)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
        msg_box.exec_()

def add_noise_Jiaoyan(self):
    if self.unit3_img3.size>1:
      SNR = 0.9
      print("Jiao Yan Ready")
      self.unit3_result2 = self.unit3_img3.copy()
      w, h = self.unit3_result2.shape[:2]
      noisy_size = int(self.unit3_result2.size*(1-SNR))
      print(noisy_size)
      for k in range(0,noisy_size):
        t = np.random.randint(0,1)
        x = int(np.random.uniform(0, w))
        y = int(np.random.uniform(0, h))
        if t<0.5:
          self.unit3_result2[x, y]=0
        else :
          self.unit3_result2[x, y]=256
      print("Jiao Yan Over")
      unit3_img_refresh(self)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
        msg_box.exec_()


def img_right_save(self):
    if self.unit3_result2.size > 1:
        fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        if fileName == '':
            return
        cv2.imwrite(fileName, self.unit3_result2)
        msg_box = QMessageBox(QMessageBox.Information, '成功', '图像保存成功,保存路径为：' + fileName)
        msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
        msg_box.exec_()


def img_right_show(self):
    if self.unit3_result2 > 1:
        cv2.imshow('Original pic', self.unit3_result2)
        cv2.waitKey(0)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '没有生成图像')
        msg_box.exec_()

def mousePressEvent(self, e):
    globalpos = e.globalPos()
    pos = self.ui.label_35.mapFromGlobal(globalpos)
    pos2 = self.ui.label_36.mapFromGlobal(globalpos)
    pos3 = self.ui.label_29.mapFromGlobal(globalpos)
    pos4 = self.ui.label_33.mapFromGlobal(globalpos)
    pos5 = self.ui.label_31.mapFromGlobal(globalpos)
    if pos.y() < 250 and pos.y() > 0 and pos.x() > 0 and pos.x() < 200:
        self.m_drag = True
        self.m_DragPosition = pos
        e.accept()
    elif pos2.y() < 250 and pos2.y() > 0 and pos2.x() > 0 and pos2.x() < 200:
        self.m_drag = True
        self.m_DragPosition = pos2
        e.accept()
    elif pos3.y() < 250 and pos3.y() > 0 and pos3.x() > 0 and pos3.x() < 200:
        self.m_drag = True
        self.m_DragPosition = pos3
        e.accept()
    elif pos4.y() < 250 and pos4.y() > 0 and pos4.x() > 0 and pos4.x() < 200:
        self.m_drag = True
        self.m_DragPosition = pos4
        e.accept()
    elif pos5.y() < 250 and pos5.y() > 0 and pos5.x() > 0 and pos5.x() < 200:
        self.m_drag = True
        self.m_DragPosition = pos5
        e.accept()
def mouseReleaseEvent(self, e):
    globalpos = e.globalPos()
    pos = self.ui.label_35.mapFromGlobal(globalpos)
    pos2 = self.ui.label_36.mapFromGlobal(globalpos)
    pos3 = self.ui.label_29.mapFromGlobal(globalpos)
    pos4 = self.ui.label_33.mapFromGlobal(globalpos)
    pos5 = self.ui.label_31.mapFromGlobal(globalpos)
    if pos.y() < 250 and pos.y() > 0 and pos.x() > 0 and pos.x() < 200:
        self.m_drag = True
        e.accept()
    elif pos2.y() < 250 and pos2.y() > 0 and pos2.x() > 0 and pos2.x() < 200:
        self.m_drag = True
        e.accept()
    elif pos3.y() < 250 and pos3.y() > 0 and pos3.x() > 0 and pos3.x() < 200:
        self.m_drag = True
        e.accept()
    elif pos4.y() < 250 and pos4.y() > 0 and pos4.x() > 0 and pos4.x() < 200:
        self.m_drag = True
        e.accept()
    elif pos5.y() < 250 and pos5.y() > 0 and pos5.x() > 0 and pos5.x() < 200:
        self.m_drag = True
        e.accept()

def mouseMoveEvent(self, e):
        e.accept()
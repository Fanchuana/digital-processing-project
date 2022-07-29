from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from math import sqrt,pow

class pltFigure(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.axes = self.fig.add_subplot(111)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


def init(self):
    self.unit5_img = np.ndarray(())
    self.unit5_imgOrg = np.ndarray(())
    self.unit5_img_channel = 1
    self.fig3 = pltFigure(width=5, height=3, dpi=80)
    self.fig_ntb3 = NavigationToolbar(self.fig3, self)
    self.gridlayout2 = QGridLayout(self.ui.label_48)
    self.gridlayout2.addWidget(self.fig3)
    self.gridlayout2.addWidget(self.fig_ntb3)
    self.unit5_img_channel=1

def hist_refresh(self):
    hist = np.bincount(self.unit5_img.ravel(), minlength=256)
    self.fig3.axes.cla()
    self.fig3.axes.plot(hist)
    self.fig3.draw()

def img_refresh(self):
    imgShow = self.unit5_img
    if self.unit5_img.size<=1:
        self.ui.label_46.setPixmap(QtGui.QPixmap(''))
        return
    h = 471
    w = 481
    M = np.float32([[1, 0, 0], [0, 1, 0]])
    h2, w2 = imgShow.shape
    print(h2)
    print(w2)
    if h2 / w2 == h / w:
        data = imgShow.tobytes()
        image = QtGui.QImage(data, w2, h2, w2 , QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(image)
        scale_pix = pix.scaled(w, h)
        self.ui.label_46.setPixmap(scale_pix)
        print("branch1")
        return
    elif h2 / w2 > h / w:
        print("branch2")
        h_ = h2
        w_ = int(h2 * w / h + 0.5)
        M[0, 2] += (w_ - w2) / 2
        M[1, 2] += (h_ - h2) / 2
        print(M)
    else:
        print("branch3")
        h_ = int(w2 * h /w  + 0.5)
        w_ = w2
        M[0, 2] += (w_ - w2) / 2
        M[1, 2] += (h_ - h2) / 2
    imgShow = cv2.warpAffine(imgShow, M, (w_, h_))
    data = imgShow.tobytes()
    image = QtGui.QImage(data, w_, h_, w_ , QtGui.QImage.Format_Grayscale8)
    pix = QtGui.QPixmap.fromImage(image)
    scale_pix = pix.scaled(w, h)
    self.ui.label_46.setPixmap(scale_pix)

def img_save(self):
    if self.unit5_img.size > 1:
        fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        if fileName == '':
            return
        cv2.imwrite(fileName, self.unit5_img)
        msg_box = QMessageBox(QMessageBox.Information, '成功', '图像保存成功,保存路径为：' + fileName)
        msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
        msg_box.exec_()

def img_clear(self):
    if self.unit5_img.size > 1:
        init(self)
        img_refresh(self)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '无需清空', '没有图片')
        msg_box.exec_()

def img_load(self):
    fileName, tmp = QFileDialog.getOpenFileName(self, '打开图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
    if fileName == '':
        return
    self.unit5_img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    self.unit5_imgOrg = self.unit5_img.copy()
    if self.unit5_img.size>= 1:
      print(self.unit5_img.shape)
    img_refresh(self)
    hist_refresh(self)

def img_show(self):
    if self.unit5_img.size > 1:
        cv2.imshow('Original pic', self.unit5_img)
        cv2.waitKey(0)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
        msg_box.exec_()


def Laplacian(self):
    if self.unit5_img.size>1:
        imarr = self.unit5_img
        height, width = imarr.shape
        fft = np.fft.fft2(imarr)
        fft = np.fft.fftshift(fft)
        for i in range(height):
            for j in range(width):
                fft[i, j] *= -((i - (height - 1) / 2) ** 2 + (j - (width - 1) / 2) ** 2)
        fft = np.fft.ifftshift(fft)
        ifft = np.fft.ifft2(fft)
        ifft = np.real(ifft)
        max = np.max(ifft)
        min = np.min(ifft)
        res = np.zeros((height, width), dtype="uint8")
        for i in range(height):
             for j in range(width):
                res[i, j] = 255 * (ifft[i, j] - min) / (max - min)
        self.unit5_img = res
        img_refresh(self)
        hist_refresh(self)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像  ')
        msg_box.exec_()


def Idea(self):
    if self.unit5_img.size>1:
      new_img = self.unit5_img
      # pencv中的傅立叶变化
      dft = cv2.dft(np.float32(new_img), flags=cv2.DFT_COMPLEX_OUTPUT)
      dtf_shift = np.fft.fftshift(dft)
      # np.fft.fftshift()函数来实现平移,让直流分量在输出图像的重心
      rows, cols = new_img.shape
      crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
      mask = np.zeros((rows, cols, 2), np.uint8)  # 生成rows行cols列的2纬矩阵，数据格式为uint8
      mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1  # 将靠近频谱中心的部分低通信息 设置为1，属于低通滤波
      fshift = dtf_shift * mask
      # 傅立叶逆变换
      f_ishift = np.fft.ifftshift(fshift)
      img_back = cv2.idft(f_ishift)
      img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # 计算像素梯度的绝对值
      img_back = np.abs(img_back)
      img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))
      self.unit5_img = (img_back * 255).astype(np.uint8)
      img_refresh(self)
      hist_refresh(self)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像  ')
        msg_box.exec_()


def Gaussion(self):
    if self.unit5_img.size > 1:
      sigma =1
      imarr = self.unit5_img
      height, width = imarr.shape
      fft = np.fft.fft2(imarr)
      fft = np.fft.fftshift(fft)
      for i in range(height):
          for j in range(width):
              fft[i, j] *= (1 - np.exp(-((i - (height - 1) / 2) ** 2 + (j - (width - 1) / 2) ** 2) / 2 / sigma ** 2))
      fft = np.fft.ifftshift(fft)
      ifft = np.fft.ifft2(fft)
      ifft = np.real(ifft)
      max = np.max(ifft)
      min = np.min(ifft)
      res = np.zeros((height, width), dtype="uint8")
      for i in range(height):
          for j in range(width):
              res[i, j] = 255 * (ifft[i, j] - min) / (max - min)
      self.unit5_img = res
      img_refresh(self)
      hist_refresh(self)
    else:
      msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像  ')
      msg_box.exec_()


#def Butterworth(self):
   #if self.unit5_img.size>1:
        #image = self.unit5_img
       # d = 40
       # f = np.fft.fft2(image)
        #fshift = np.fft.fftshift(f)
       # transfor_matrix = np.zeros(image.shape)
       # M = transfor_matrix.shape[0]
       # N = transfor_matrix.shape[1]
     #   for u in range(M):
          #  for v in range(N):
             #   D = sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
               # transfor_matrix[u, v] = 1 / (1 + pow(D / d, 16))
      #  print(transfor_matrix.dtype)
       # new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * transfor_matrix)))
       # new_img = new_img .astype(np.float32)
       # self.unit5_img = (new_img * 255).astype(np.uint8)
      #  self.unit5_img = new_img
      # print(self.unit5_img.shape)
       # img_refresh(self)
      #  #hist_refresh(self)
   # else:
      #  msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请先选择一副图像  ')
      #  msg_box.exec_()


def img_reset(self):
    if self.unit5_img.size>1:
        self.unit5_img = self.unit5_imgOrg
        img_refresh(self)
        hist_refresh(self)
    else:
        msgbox = QMessageBox(QMessageBox.Warning, "提示", "请选择图像")
        msgbox.exec_()

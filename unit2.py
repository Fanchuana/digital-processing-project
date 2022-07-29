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
        self.img2 = np.ndarray(())
        self.img2Org = np.ndarray(())
        self.img2Show = np.ndarray(())
        self.channel2 = 1
        self.fig1 = pltFigure(width=5, height=3, dpi=80)
        self.fig_ntb1 = NavigationToolbar(self.fig1, self)
        self.gridlayout1 = QGridLayout(self.ui.label_18)
        self.gridlayout1.addWidget(self.fig1)
        self.gridlayout1.addWidget(self.fig_ntb1)

def reinit(self):
    self.img2 = np.ndarray(())
    self.img2Org = np.ndarray(())
    self.img2Show = np.ndarray(())
    self.channel2 = 1

def unit2_img_load(self):
        fileName, tmp = QFileDialog.getOpenFileName(self, '打开图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        reinit(self)
        if fileName == '':
            return
        self.img2 = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
        self.img2Org = self.img2.copy()
        if self.img2.size <= 1:
            return
        self.gMean, self.gStd = cv2.meanStdDev(self.img2)
        self.gMean = round(self.gMean[0][0], 3)
        self.gStd = round(self.gStd[0][0], 3)
        self.h2, self.w2 = self.img2.shape
        print(self.img2.shape)
        unit2_img_refresh(self)

def unit2_img_reset(self):
        if self.img2.size > 1:
            temp_img = self.img2Org
            reinit(self)
            self.img2 = temp_img
            self.gMean, self.gStd = cv2.meanStdDev(self.img2)
            self.gMean = round(self.gMean[0][0], 3)
            self.gStd = round(self.gStd[0][0], 3)
            self.h2, self.w2 = self.img2.shape
            self.img2Org = self.img2.copy()
            unit2_img_refresh(self)
            hist = np.bincount(self.img2.ravel(), minlength=256)
            hist_refresh(self, hist)
        else:
            msgbox = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
            msgbox.exec_()

def unit2_img_showNew(self):
    if self.img2.size > 1:
        cv2.imshow('Original pic', self.img2)
        cv2.waitKey(0)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
        msg_box.exec_()


def unit2_img_save(self):
    if self.img2.size > 1:
        fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        if fileName == '':
            return
        cv2.imwrite(fileName, self.img2)
        msg_box = QMessageBox(QMessageBox.Information, '成功', '图像保存成功,保存路径为：' + fileName)
        msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
        msg_box.exec_()

def hist_refresh(self,hist):
    self.fig1.axes.cla()
    self.fig1.axes.plot(hist)
    self.fig1.draw()

def clahe(self):
    if self.img2.size>1:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            self.img2 = clahe.apply(self.img2Org)
            unit2_img_refresh(self)
            hist = np.bincount(self.img2.ravel(), minlength=256)
            hist_refresh(self, hist)
    else:
            msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
            msg_box.exec_()

def eqHist(self):
    if self.img2.size > 1:
         self.img2 = cv2.equalizeHist(self.img2Org)
         unit2_img_refresh(self)
         hist = np.bincount(self.img2.ravel(), minlength=256)
         hist_refresh(self, hist)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
        msg_box.exec_()


def globalH(self):
    if self.img2.size>1:
        hist = np.bincount(self.img2.ravel(), minlength=256)
        hist_refresh(self,hist)
        Mean, Std = cv2.meanStdDev(self.img2)
        Mean = round(Mean[0][0], 3)
        Std = round(Std[0][0], 3)
        self.ui.textBrowser_2.setText('%s' % Mean)
        self.ui.textBrowser_5.setText('%s' % Std)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像  ')
        msg_box.exec_()

def localH(self):
    x1 = self.ui.lineEdit9.text()
    x2 = self.ui.lineEdit10.text()
    y1 = self.ui.lineEdit_15.text()
    y2 = self.ui.lineEdit_16.text()
    if x1 and x2 and y1 and y2:
        try:
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            a1 = x1 if x1>x2 else x2
            a2 = x1 if x1<=x2 else x2
            b1 = y1 if y1>y2 else y2
            b2 = y1 if y1<=y2 else y2
            img = self.img2[a2:a1+1,b2:b1+1]
            hist = np.bincount(img.ravel(), minlength=256)
            hist_refresh(self, hist)
            Mean, Std = cv2.meanStdDev(img)
            Mean = round(Mean[0][0], 3)
            Std = round(Std[0][0],3)
            self.ui.textBrowser_2.setText('%s' % Mean)
            self.ui.textBrowser_5.setText('%s' % Std)
        except:
            msg_box = QMessageBox(QMessageBox.Warning, '参数异常', '请重新输入参数  ')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像  ')
        msg_box.exec_()



def unit2_clear(self):
    if self.img2.size>1:
        init(self)
        self.ui.label_16.setPixmap(QtGui.QPixmap(''))
        msg_box = QMessageBox(QMessageBox.Information, '清空完成', '可以重新添加图片  ')
        msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图片', '无需清除  ')
        msg_box.exec_()

def unit2_img_refresh(self):
        self.img2Show = self.img2
        M = np.float32([[1, 0, 0], [0, 1, 0]])
        if self.h2 / self.w2 == 360 / 550:
            data = self.img2Show.tobytes()
            image = QtGui.QImage(data, self.w2, self.h2, self.w2 * self.channel2, QtGui.QImage.Format_Grayscale8)
            pix = QtGui.QPixmap.fromImage(image)
            scale_pix = pix.scaled(550, 360)
            self.ui.label_16.setPixmap(scale_pix)
            return
        elif self.h2 / self.w2 > 360 / 550:
            h_ = self.h2
            w_ = int(self.h2 * 550 / 360 + 0.5)
            M[0, 2] += (w_ - self.w2) / 2
            M[1, 2] += (h_ - self.h2) / 2
            print(M)
        else:
            h_ = int(self.w2 * 360 / 550 + 0.5)
            w_ = self.w2
            M[0, 2] += (w_ - self.w2) / 2
            M[1, 2] += (h_ - self.h2) / 2
        self.img2Show = cv2.warpAffine(self.img2Show, M, (w_, h_))
        data = self.img2Show.tobytes()
        image = QtGui.QImage(data, w_, h_, w_ * self.channel2, QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(image)
        scale_pix = pix.scaled(550, 360)
        self.ui.label_16.setPixmap(scale_pix)

def mouseReleaseEvent(self, e):
    if self.img2.size > 1:
       globalpos = e.globalPos()
       pos = self.ui.label_16.mapFromGlobal(globalpos)
       if pos.y() < 360 and pos.y() > 0 and pos.x() > 0 and pos.x() < 550:
           self.m_drag = False
           e.accept()
       else:
           e.accept()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像  ')
        msg_box.exec_()


def mousePressEvent(self, e):
    if self.img2.size > 1:
       globalpos = e.globalPos()
       pos = self.ui.label_16.mapFromGlobal(globalpos)
       if pos.y() < 360 and pos.y() > 0 and pos.x() > 0 and pos.x() < 550:
           self.m_drag = True
           self.m_DragPosition = pos
           e.accept()
       else:
           e.accept()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像  ')
        msg_box.exec_()


def mouseMoveEvent(self, e):
    globalpos = e.globalPos()
    pos = self.ui.label_16.mapFromGlobal(globalpos)
    if pos.y() < 360 and pos.y() > 0 and pos.x() > 0 and pos.x() < 550:
        h = self.img2.shape[0]
        w = self.img2.shape[1]
        self.ui.lineEdit9.setText('%s' % round(self.m_DragPosition.x()/550*w))
        self.ui.lineEdit10.setText('%s' % round(pos.x()/550*w))
        self.ui.lineEdit_15.setText('%s' % round(self.m_DragPosition.y()/ 360 * h))
        self.ui.lineEdit_16.setText('%s' % round(pos.y()/ 360 * h))
        e.accept()
    else:
        e.accept()


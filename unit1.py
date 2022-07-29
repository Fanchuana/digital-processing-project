from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import cv2
import numpy as np
import os

#
def unit1_img_load(self):
    fileName, tmp = QFileDialog.getOpenFileName(self, '打开图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
    if fileName == '':
       return
    root_dir, file_name = os.path.split(fileName)
    self.img = cv2.imread(fileName, -1)
    if self.img.size <= 1:
        return
    self.fname = file_name.split('.')[0]
    self.imgOrg = self.img.copy()
    if len(self.img.shape) == 3:
        self.channel = 3
        if self.img.shape[2] == 4:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGRA2BGR)
    print(self.img.shape)
    img_refresh(self)





def unit1_img_reset(self):
    if self.img.size>1:
        self.img = self.imgOrg
        img_refresh(self)
    else:
        msgbox = QMessageBox(QMessageBox.Warning, "没有图像", "请选择图像")
        msgbox.exec_()



def unit1_img_show(self):
    if self.img.size > 1:
        cv2.imshow('Original pic', self.img)
        cv2.waitKey(0)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像  ')
        msg_box.exec_()


def unit1_img_clear(self):
    self.img = np.ndarray(())
    self.imgOrg = np.ndarray(())
    self.imgShow = np.ndarray(())
    self.fname = ''
    self.w = 0
    self.h = 0
    self.c = 1
    self.ui.textBrowser.setText('')
    self.ui.textBrowser_3.setText('')
    self.ui.textBrowser_4.setText('')
    self.ui.label_10.setPixmap(QtGui.QPixmap(''))

def unit1_img_save(self):
    if self.img.size>1:
        fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        if fileName == '':
            return
        cv2.imwrite(fileName, self.img)
        msg_box = QMessageBox(QMessageBox.Information, '成功', '图像保存成功,保存路径为：'+fileName)
        msg_box.exec_()

    else:
        msg_box = QMessageBox(QMessageBox.Warning, '没有图像', '请选择图像')
        msg_box.exec_()


def trans_by_rate(self):
    if self.img.size>1:
        x=self.ui.lineEdit.text()
        y=self.ui.lineEdit_2.text()
        if x and y:
            x=float(x)/100
            y=float(y)/100
            M = np.float32([[1, 0, x * self.w], [0, 1, y * self.h]])
            self.img = cv2.warpAffine(self.img, M, (self.w, self.h))
            img_refresh(self)
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '提示', '请正确输入')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
        msg_box.exec_()

def trans_by_pixel(self):
    if self.img.size > 1:
        x = self.ui.lineEdit_3.text()
        y = self.ui.lineEdit_4.text()
        if x and y:
            try:
               x = int(x)
               y = int(y)
               M = np.float32([[1, 0, x], [0, 1, y]])
               self.img = cv2.warpAffine(self.img, M, (self.w, self.h))
               img_refresh(self)
            except:
                msg_box = QMessageBox(QMessageBox.Warning, '异常', '请输入整数')
                msg_box.exec_()
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '提示', '请正确输入')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
        msg_box.exec_()


def scale_by_rate(self):
    if self.img.size>1:
         scale = self.ui.lineEdit_10.text()
         if scale:
            print("YES")
            scale = float(scale)/100
            print(scale)
            x = int(self.w * scale)
            y = int(self.h * scale)
            print(x)
            self.img = cv2.resize(self.img, (x, y))
            img_refresh(self)
         else:
            msg_box = QMessageBox(QMessageBox.Warning, '提示', '请正确输入')
            msg_box.exec_()
    else:
          msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
          msg_box.exec_()

def rotate(self):
    if self.img.size > 1:
        theta = self.ui.lineEdit_9.text()
        if theta:
            try:
                theta = float(theta)
                M = cv2.getRotationMatrix2D((self.h / 2, self.w / 2), theta, 1)
                self.img = cv2.warpAffine(self.img, M, (self.w, self.h))
                img_refresh(self)
            except:
                msg_box = QMessageBox(QMessageBox.Warning, '异常', '旋转异常，请检查参数')
                msg_box.exec_()
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '提示', '请正确输入')
            msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
        msg_box.exec_()


def affine_trans(self):
    if self.img.size>1:
       x1 = self.ui.lineEdit_5.text()
       x2 = self.ui.lineEdit_6.text()
       y1 = self.ui.lineEdit_7.text()
       y2 = self.ui.lineEdit_8.text()
       z1 = self.ui.lineEdit_43.text()
       z2 = self.ui.lineEdit_44.text()
       if x1 and x2 and y1 and y2 and z1 and z2:
           try:
               x1 = float(x1)
               y1 = float(y1)
               z1 = float(z1)
               x2 = float(x2)
               y2 = float(y2)
               z2 = float(z2)
               M = np.float32([[x1, y1, z1], [x2, y2, z2]])
               self.img = cv2.warpAffine(self.img, M, (self.w, self.h))
               img_refresh(self)
           except:
               msg_box = QMessageBox(QMessageBox.Warning, '异常', '仿射变换异常，请检查参数')
               msg_box.exec_()
       else:
           msg_box = QMessageBox(QMessageBox.Warning, '提示', '请正确输入')
           msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图像')
        msg_box.exec_()

def img_refresh(self):
    self.imgShow = self.img
    self.h = self.imgShow.shape[0]
    self.w = self.imgShow.shape[1]
    self.ui.textBrowser.setText('%s×%s×%s' % (self.w, self.h, self.channel))
    M = np.float32([[1, 0, 0], [0, 1, 0]])
    # print(M)
    if self.h / self.w == 50/72:
        data = self.imgShow.tobytes()
        if self.channel == 3:
            image = QtGui.QImage(data, self.w, self.h, self.w * self.channel, QtGui.QImage.Format_BGR888)
        else:
            image = QtGui.QImage(data, self.w, self.h, self.w * self.channel, QtGui.QImage.Format_Grayscale8)

        w_label = self.ui.label_10.width()
        h_label = self.ui.label_10.height()
        pix = QtGui.QPixmap.fromImage(image)
        scale_pix = pix.scaled(w_label, h_label)
        self.ui.label_10.setPixmap(scale_pix)
        return
    elif self.h / self.w > 50 / 72:
        h_ = self.h
        w_ = int(self.h * 72 / 50 + 0.5)
        M[0, 2] += (w_ - self.w) / 2
        M[1, 2] += (h_ - self.h) / 2
    else:
        h_ = int(self.w * 50 / 72 + 0.5)
        w_ = self.w
        M[0, 2] += (w_ - self.w) / 2
        M[1, 2] += (h_ - self.h) / 2
    self.imgShow = cv2.warpAffine(self.imgShow, M, (w_, h_))
    data = self.imgShow.tobytes()
    if self.channel == 3:
        image = QtGui.QImage(data, w_, h_, w_ * self.channel, QtGui.QImage.Format_BGR888)
    else:
        image = QtGui.QImage(data, w_, h_, w_ * self.channel, QtGui.QImage.Format_Grayscale8)

    w_label = self.ui.label_10.width()
    h_label = self.ui.label_10.height()
    pix = QtGui.QPixmap.fromImage(image)
    scale_pix = pix.scaled(w_label, h_label)
    self.ui.label_10.setPixmap(scale_pix)


def mouseReleaseEvent(self, e):
        if self.imgShow.size > 1:
            h = self.imgShow.shape[0]
            w = self.imgShow.shape[1]
            c = self.channel
            globalpos = e.globalPos()
            pos = self.ui.label_10.mapFromGlobal(globalpos)
            if pos.y() < 500 and pos.y() > 0 and pos.x() > 0 and pos.x() < 720:
                x = int(pos.x() / 720 * w)
                y = int(pos.y() / 500 * h)
                self.ui.textBrowser_4.setText(' (%s, %s)' % (x, y))
                if c == 3:
                    rgb = self.imgShow[y, x]
                    self.ui.textBrowser_3.setText(' R%s G%s B%s' % (rgb[2], rgb[1], rgb[0]))
                else:
                    gray = self.imgShow[y, x]
                    self.ui.textBrowser_3.setText(' G %s' % gray)
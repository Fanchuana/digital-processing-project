from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import cv2
import numpy as np



def init(self):
    self.unit4_img = np.ndarray(())
    self.unit4_robertsimg = np.ndarray(())
    self.unit4_prewittimg = np.ndarray(())
    self.unit4_logimg = np.ndarray(())
    self.unit4_sobelimg = np.ndarray(())
    self.unit4_laplacianimg = np.ndarray(())
    self.unit4_loughimg = np.ndarray(())
    self.unit4_cannyimg = np.ndarray(())
    self.unit4_img_channel = 1
    self.unit4_robertsimg_channel = 1
    self.unit4_prewittimg_channel = 1
    self.unit4_logimg_channel = 1
    self.unit4_sobelimg_channel = 1
    self.unit4_laplacianimg_channel = 1
    self.unit4_loughimg_channel = 1
    self.unit4_cannyimg_channel = 1

def img_load(self):
    fileName, tmp = QFileDialog.getOpenFileName(self, '打开图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
    if fileName == '':
        return
    init(self)
    self.unit4_img = cv2.imread(fileName, -1)
    if self.unit4_img.size <= 1:
        return
    if len(self.unit4_img.shape) == 3:
        self.unit4_img_channel = 3
        if self.unit4_img.shape[2] == 4:
            self.unit4_img = cv2.cvtColor(self.unit4_img, cv2.COLOR_BGRA2BGR)
    print(self.unit4_img.shape)
    unit4_img_refresh(self)


def unit4_img_refresh(self):
    array = \
    [self.unit4_img,
    self.unit4_robertsimg,
    self.unit4_prewittimg,
    self.unit4_logimg,
    self.unit4_sobelimg,
    self.unit4_laplacianimg,
    self.unit4_loughimg,
    self.unit4_cannyimg]

    array2 = [self.ui.label_53,
              self.ui.label_43,
              self.ui.label_30,
              self.ui.label_49,
              self.ui.label_58,
              self.ui.label_32,
              self.ui.label_34,
              self.ui.label_50]

    channel = [self.unit4_img_channel,
               self.unit4_robertsimg_channel,
               self.unit4_prewittimg_channel,
               self.unit4_logimg_channel,
               self.unit4_sobelimg_channel,
               self.unit4_laplacianimg_channel,
               self.unit4_loughimg_channel,
               self.unit4_cannyimg_channel]
    height = 240
    weight = 240
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
            w_ = int(index_h * weight / height+ 0.5)
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
def img_clear(self):
    if self.unit4_img.size > 1:
        init(self)
        unit4_img_refresh(self)
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '无需清空', '没有图片')
        msg_box.exec_()


def action(self):
    if self.unit4_img.size>1:
        #try:
            Roberts(self)
            print("Roberts over!")
            Prewitt(self)
            print("Prewitt over!")
            Log(self)
            print("Log over!")
            Sobel(self)
            print("Sobel over!")
            Laplacian(self)
            print("Laplacian over!")
            Lough(self)
            print("Lough over!")
            Canny(self)
            print("Canny over!")
            unit4_img_refresh(self)
        #except:
            #msg_box = QMessageBox(QMessageBox.Warning, '算子执行异常', '请更换图片')
            #msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '请选择图片')
        msg_box.exec_()



def Roberts(self):
    img = self.unit4_img
    if(self.unit4_img_channel == 3):
       img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #  Roberts算子
    kernelx = np.array([[-1,0],[0,1]],dtype=int)
    kernely = np.array([[0,-1],[1,0]],dtype=int)
    #  卷积操作
    x = cv2.filter2D(img,cv2.CV_16S,kernelx)
    y = cv2.filter2D(img,cv2.CV_16S,kernely)
    #  数据格式转换
    absX=cv2.convertScaleAbs(x)
    absY=cv2.convertScaleAbs(y)
    self.unit4_robertsimg = cv2.addWeighted(absX,0.5,absY,0.5,0)

def Prewitt(self):
    grayImage = self.unit4_img
    if (self.unit4_img_channel == 3):
       grayImage = cv2.cvtColor(grayImage, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)

    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    self.unit4_prewittimg = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

def Log(self):
    img = self.unit4_img
    if (self.unit4_img_channel == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2边缘扩充处理图像并使用高斯滤波处理该图像
    image = cv2.copyMakeBorder(img, 2, 2, 2, 2, borderType=cv2.BORDER_REPLICATE)
    image = cv2.GaussianBlur(image, (3, 3), 0, 0)
    # 使用Numpy定义LoG算子
    m1 = np.array(
        [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
    # 卷积运算
    rows = image.shape[0]
    cols = image.shape[1]
    image1 = np.zeros(image.shape)
    for i in range(2, rows - 2):
            for j in range(2, cols - 2):
                image1[i, j] = np.sum((m1 * image[i - 2:i + 3, j - 2:j + 3]))
    self.unit4_logimg = cv2.convertScaleAbs(image1)

def Sobel(self):
    grayImage = self.unit4_img
    if (self.unit4_img_channel == 3):
       grayImage = cv2.cvtColor(grayImage, cv2.COLOR_BGR2GRAY)
    # 求Sobel 算子
    x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)
    # 数据格式转换
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    self.unit4_sobelimg = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

def Laplacian(self):
    grayImage = self.unit4_img
    if (self.unit4_img_channel == 3):
       grayImage = cv2.cvtColor(grayImage, cv2.COLOR_BGR2GRAY)
    #  高斯滤波
    grayImage = cv2.GaussianBlur(grayImage,ksize=(5,5),sigmaX=0,sigmaY=0)
    #  拉普拉斯算法
    dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
    # 数据格式转换
    self.unit4_laplacianimg = cv2.convertScaleAbs(dst)

def Lough(self):
    img = self.unit4_img
    if (self.unit4_img_channel == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 2, 118)
    result = img.copy()
    for i_line in lines:
        for line in i_line:
            rho = line[0]
            theta = line[1]
            if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                pt1 = (int(rho / np.cos(theta)), 0)
                pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                cv2.line(result, pt1, pt2, (0, 0, 255))
            else:
                pt1 = (0, int(rho / np.sin(theta)))
                pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                cv2.line(result, pt1, pt2, (0, 0, 255), 1)
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, 200, 15)
    result_P = img.copy()
    for i_P in linesP:
        for x1, y1, x2, y2 in i_P:
            cv2.line(result_P, (x1, y1), (x2, y2), (0, 255, 0), 3)
    self.unit4_loughimg = result_P
def Canny(self):
    image = self.unit4_img
    if (self.unit4_img_channel == 3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #  高斯滤波
    image= cv2.GaussianBlur(image, (3, 3), 0)
    #  求x，y方向的Sobel算子
    gradx = cv2.Sobel(image, cv2.CV_16SC1, 1, 0)
    grady = cv2.Sobel(image, cv2.CV_16SC1, 0, 1)
    print("Yes")
    #  使用Canny函数处理图像，x,y分别是3求出来的梯度，低阈值50，高阈值150
    self.unit4_cannyimg = cv2.Canny(gradx, grady, 50, 150)




def canny_save(self):
    if self.unit4_cannyimg.size > 1:
        fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        if fileName == '':
            return
        cv2.imwrite(fileName, self.unit4_cannyimg)
        msg_box = QMessageBox(QMessageBox.Information, '成功', '图像保存成功,保存路径为：' + fileName)
        msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '没有生成图像')
        msg_box.exec_()


def log_save(self):
    if self.unit4_logimg.size > 1:
        fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        if fileName == '':
            return
        cv2.imwrite(fileName, self.unit4_logimg)
        msg_box = QMessageBox(QMessageBox.Information, '成功', '图像保存成功,保存路径为：' + fileName)
        msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '没有生成图像')
        msg_box.exec_()


def lough_save(self):
    if self.unit4_loughimg.size > 1:
        fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        if fileName == '':
            return
        cv2.imwrite(fileName, self.unit4_loughimg)
        msg_box = QMessageBox(QMessageBox.Information, '成功', '图像保存成功,保存路径为：' + fileName)
        msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '没有生成图像')
        msg_box.exec_()


def laplacian_save(self):
    if self.unit4_laplacianimg.size > 1:
        fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        if fileName == '':
            return
        cv2.imwrite(fileName, self.unit4_laplacianimg)
        msg_box = QMessageBox(QMessageBox.Information, '成功', '图像保存成功,保存路径为：' + fileName)
        msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '没有生成图像')
        msg_box.exec_()


def sobel_save(self):
    if self.unit4_sobelimg.size > 1:
        fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        if fileName == '':
            return
        cv2.imwrite(fileName, self.unit4_laplacianimg)
        msg_box = QMessageBox(QMessageBox.Information, '成功', '图像保存成功,保存路径为：' + fileName)
        msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '没有生成图像')
        msg_box.exec_()


def prewitt_save(self):
    if self.unit4_prewittimg.size > 1:
        fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        if fileName == '':
            return
        cv2.imwrite(fileName, self.unit4_prewittimg)
        msg_box = QMessageBox(QMessageBox.Information, '成功', '图像保存成功,保存路径为：' + fileName)
        msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '没有生成图像')
        msg_box.exec_()


def roberts_save(self):
    if self.unit4_robertsimg.size > 1:
        fileName, tmp = QFileDialog.getSaveFileName(self, '保存图像', 'Image', '*.png *.jpg *.bmp *.jpeg')
        if fileName == '':
            return
        cv2.imwrite(fileName, self.unit4_robertsimg)
        msg_box = QMessageBox(QMessageBox.Information, '成功', '图像保存成功,保存路径为：' + fileName)
        msg_box.exec_()
    else:
        msg_box = QMessageBox(QMessageBox.Warning, '提示', '没有生成图像')
        msg_box.exec_()

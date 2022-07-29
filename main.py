import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.Qt import QApplication, QWidget, QThread
import numpy as np
import ui
import unit1
import unit2
import unit3
import unit4
import unit5
import unit6
import unit7


class MainDialog(QMainWindow):
    def __init__(self, parent=None):
        super(MainDialog, self).__init__(parent)
        self.ui = ui.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('CV lrioxh')
        self.m_drag = False
        self.img = np.ndarray(())
        self.imgOrg=np.ndarray(())
        self.imgShow = np.ndarray(())
        self.fname =''
        self.w=0
        self.h=0
        self.c=1
        self.ui.pushButton.clicked.connect(self.unit1_img_load) #第一个pushbutton 用来选择场景一的图像
        self.ui.pushButton_2.clicked.connect(self.unit1_img_reset) #用来重置图像
        self.ui.pushButton_3.clicked.connect(self.unit1_img_show)
        self.ui.pushButton_4.clicked.connect(self.unit1_img_clear)
        self.ui.pushButton_5.clicked.connect(self.unit1_img_save)
        self.ui.pushButton_6.clicked.connect(self.trans_by_rate)
        self.ui.pushButton_7.clicked.connect(self.trans_by_pixel)
        self.ui.pushButton_8.clicked.connect(self.rotate)
        self.ui.pushButton_9.clicked.connect(self.scale_by_rate)
        self.ui.pushButton_10.clicked.connect(self.affine_trans)

        unit2.init(self)
        self.ui.pushButton_11.clicked.connect(self.unit2_img_load)
        self.ui.pushButton_12.clicked.connect(self.unit2_globalH)
        self.ui.pushButton_13.clicked.connect(self.unit2_localH)
        self.ui.pushButton_14.clicked.connect(self.unit2_img_showNew)
        self.ui.pushButton_15.clicked.connect(self.unit2_eqHist)
        self.ui.pushButton_16.clicked.connect(self.unit2_clahe)
        self.ui.pushButton_17.clicked.connect(self.unit2_clear)
        self.ui.pushButton_18.clicked.connect(self.unit2_img_reset)
        self.ui.pushButton_47.clicked.connect(self.unit2_img_save)

        unit3.init(self)
        self.ui.pushButton_64.clicked.connect(self.unit3_img_left_load1)
        self.ui.pushButton_66.clicked.connect(self.unit3_img_left_load2)
        self.ui.pushButton_21.clicked.connect(self.unit3_ADD)
        self.ui.pushButton_19.clicked.connect(self.unit3_SUB)
        self.ui.pushButton_24.clicked.connect(self.unit3_MULTI)
        self.ui.pushButton_22.clicked.connect(self.unit3_DIVIDE)
        self.ui.pushButton_20.clicked.connect(self.unit3_AND)
        self.ui.pushButton_48.clicked.connect(self.unit3_OR)
        self.ui.pushButton_49.clicked.connect(self.unit3_NOT)
        self.ui.pushButton_57.clicked.connect(self.unit3_img_left_clear)
        self.ui.pushButton_61.clicked.connect(self.unit3_img_left_save)
        self.ui.pushButton_62.clicked.connect(self.unit3_img_left_show)
        self.ui.pushButton_65.clicked.connect(self.unit3_img_left_load)
        self.ui.pushButton_50.clicked.connect(self.unit3_erode)
        self.ui.pushButton_51.clicked.connect(self.unit3_dilate)
        self.ui.pushButton_52.clicked.connect(self.unit3_opening)
        self.ui.pushButton_53.clicked.connect(self.unit3_closing)
        self.ui.pushButton_55.clicked.connect(self.unit3_mean)
        self.ui.pushButton_23.clicked.connect(self.unit3_guassian)
        self.ui.pushButton_54.clicked.connect(self.unit3_Covfilter)
        self.ui.pushButton_56.clicked.connect(self.unit3_median)
        self.ui.pushButton_58.clicked.connect(self.unit3_img_right_clear)
        self.ui.pushButton_59.clicked.connect(self.unit3_img_right_save)
        self.ui.pushButton_63.clicked.connect(self.unit3_img_right_show)
        self.ui.pushButton_67.clicked.connect(self.unit3_add_noise_Guass)
        self.ui.pushButton_68.clicked.connect(self.unit3_add_noise_Jiaoyan)
        self.ui.pushButton_60.clicked.connect(self.unit3_bilateralFilter)

        unit4.init(self)
        self.ui.pushButton_25.clicked.connect(self.unit4_img_load)
        self.ui.pushButton_26.clicked.connect(self.unit4_action)
        self.ui.pushButton_30.clicked.connect(self.unit4_img_clear)
        self.ui.pushButton_27.clicked.connect(self.unit4_roberts_save)
        self.ui.pushButton_28.clicked.connect(self.unit4_prewitt_save)
        self.ui.pushButton_29.clicked.connect(self.unit4_sobel_save)
        self.ui.pushButton_69.clicked.connect(self.unit4_laplacian_save)
        self.ui.pushButton_70.clicked.connect(self.unit4_lough_save)
        self.ui.pushButton_44.clicked.connect(self.unit4_log_save)
        self.ui.pushButton_45.clicked.connect(self.unit4_canny_save)

        unit5.init(self)
        self.ui.pushButton_31.clicked.connect(self.unit5_img_load)
        self.ui.pushButton_39.clicked.connect(self.unit5_img_clear)
        self.ui.pushButton_37.clicked.connect(self.unit5_img_save)
        self.ui.pushButton_40.clicked.connect(self.unit5_img_reset)
       # self.ui.pushButton_32.clicked.connect(self.unit5_Butterworth)
        self.ui.pushButton_35.clicked.connect(self.unit5_Gaussion)
        self.ui.pushButton_33.clicked.connect(self.unit5_Idea)
        self.ui.pushButton_36.clicked.connect(self.unit5_Laplacian)
        self.ui.pushButton_46.clicked.connect(self.unit5_img_show)

        unit6.init(self)
        self.ui.pushButton_34.clicked.connect(self.unit6_img_load1)
        self.ui.pushButton_38.clicked.connect(self.unit6_img_clear)
        self.ui.pushButton_41.clicked.connect(self.unit6_img_load2)
        self.ui.pushButton_43.clicked.connect(self.unit6_img_save)
        self.ui.pushButton_42.clicked.connect(self.unit6_style_transfer)

        unit7.init(self)
        self.ui.pushButton_32.clicked.connect(self.unit7_img_load)
        self.ui.pushButton_71.clicked.connect(self.unit7_model_load)
        self.ui.pushButton_72.clicked.connect(self.unit7_object_detection)
        self.ui.pushButton_73.clicked.connect(self.unit7_result_show)
        self.ui.pushButton_74.clicked.connect(self.unit7_clear)
        self.ui.pushButton_75.clicked.connect(self.unit7_result_save)

    ##unit7
    def unit7_result_save(self):
        return unit7.result_save(self)


    def unit7_clear(self):
        return unit7.clear(self)


    def unit7_result_show(self):
        return unit7.result_show(self)

    def unit7_object_detection(self):
        return unit7.object_detection(self)

    def unit7_model_load(self):
        return unit7.model_load(self)

    def unit7_img_load(self):
        return unit7.img_load(self)



    ##unit6
    def unit6_style_transfer(self):
        return unit6.style_transfer(self)


    def unit6_img_save(self):
        return unit6.img_save(self)


    def unit6_img_load2(self):
        return unit6.img_load2(self)


    def unit6_img_clear(self):
        return unit6.img_clear(self)

    def unit6_img_load1(self):
        return unit6.img_load1(self)


    ##unit5
    def unit5_img_show(self):
        return unit5.img_show(self)

    def unit5_Laplacian(self):
        return unit5.Laplacian(self)

    def unit5_Idea(self):
        return unit5.Idea(self)

    def unit5_Gaussion(self):
        return unit5.Gaussion(self)

  #  def unit5_Butterworth(self):
     #   return unit5.Butterworth(self)

    def unit5_img_reset(self):
        return unit5.img_reset(self)

    def unit5_img_save(self):
        return unit5.img_save(self)

    def unit5_img_clear(self):
        return unit5.img_clear(self)

    def unit5_img_load(self):
        return unit5.img_load(self)


    ##unit4
    def unit4_canny_save(self):
        return unit4.canny_save(self)

    def unit4_log_save(self):
        return unit4.log_save(self)

    def unit4_lough_save(self):
        return unit4.lough_save(self)

    def unit4_laplacian_save(self):
        return unit4.laplacian_save(self)

    def unit4_sobel_save(self):
        return unit4.sobel_save(self)

    def unit4_prewitt_save(self):
        return unit4.prewitt_save(self)

    def unit4_roberts_save(self):
        return unit4.roberts_save(self)

    def unit4_img_clear(self):
        return unit4.img_clear(self)

    def unit4_action(self):
        return unit4.action(self)

    def unit4_img_load(self):
        return unit4.img_load(self)

    ##unit3
    def unit3_bilateralFilter(self):
        return unit3.bilateralFilter(self)

    def unit3_add_noise_Jiaoyan(self):
        return unit3.add_noise_Jiaoyan(self)

    def unit3_add_noise_Guass(self):
        return unit3.add_noise_Guass(self)

    def unit3_img_left_load1(self):
        return unit3.img_left_load1(self)

    def unit3_img_left_load2(self):
        return unit3.img_left_load2(self)

    def unit3_ADD(self):
        return unit3.ADD(self)

    def unit3_SUB(self):
        return unit3.SUB(self)

    def unit3_MULTI(self):
        return unit3.MULTI(self)

    def unit3_DIVIDE(self):
        return unit3.DIVIDE(self)

    def unit3_AND(self):
        return unit3.AND(self)

    def unit3_OR(self):
        return unit3.OR(self)

    def unit3_NOT(self):
        return unit3.NOT(self)

    def unit3_img_left_clear(self):
        return unit3.img_left_clear(self)

    def unit3_img_left_save(self):
        return unit3.img_left_save(self)

    def unit3_img_left_show(self):
        return unit3.img_left_show(self)

    def unit3_img_left_load(self):
        return unit3.img_right_load(self)

    def unit3_erode(self):
        return unit3.erode(self)

    def unit3_dilate(self):
        return unit3.dilate(self)

    def unit3_opening(self):
        return unit3.opening(self)

    def unit3_closing(self):
        return unit3.closing(self)

    def unit3_mean(self):
        return unit3.mean(self)

    def unit3_guassian(self):
        return unit3.guassian(self)

    def unit3_Covfilter(self):
        return unit3.Covfilter(self)

    def unit3_median(self):
        return unit3.median(self)

    def unit3_img_right_clear(self):
        return unit3.img_right_clear(self)

    def unit3_img_right_save(self):
        return unit3.img_right_save(self)

    def unit3_img_right_show(self):
        return unit3.img_right_show(self)
        ### U2

    def unit2_img_load(self):
        return unit2.unit2_img_load(self)

    def unit2_img_reset(self):
        return unit2.unit2_img_reset(self)

    def unit2_img_showNew(self):
        return unit2.unit2_img_showNew(self)

    def unit2_img_save(self):
        return unit2.unit2_img_save(self)

    def unit2_clahe(self):
        return unit2.clahe(self)

    def unit2_eqHist(self):
        return unit2.eqHist(self)

    def unit2_globalH(self):
        return unit2.globalH(self)

    def unit2_localH(self):
        return unit2.localH(self)

    def unit2_enhance(self):
        return unit2.enhance(self)

    def unit2_clear(self):
        return unit2.unit2_clear(self)

    ### U1
    def unit1_img_load(self):
        return unit1.unit1_img_load(self)

    def unit1_img_reset(self):
        return unit1.unit1_img_reset(self)

    def unit1_img_show(self):
        return unit1.unit1_img_show(self)

    def unit1_img_clear(self):
        return unit1.unit1_img_clear(self)

    def unit1_img_save(self):
        return unit1.unit1_img_save(self)

    def trans_by_rate(self):
        return unit1.trans_by_rate(self)

    def trans_by_pixel(self):
        return unit1.trans_by_pixel(self)

    def scale_by_rate(self):
        return unit1.scale_by_rate(self)

    def rotate(self):
        return unit1.rotate(self)

    def affine_trans(self):
        return unit1.affine_trans(self)







    ###
    def mouseReleaseEvent(self, e):
        if Qt.LeftButton:
            if self.ui.tabWidget.currentIndex() == 0:
                return unit1.mouseReleaseEvent(self,e)
            if self.ui.tabWidget.currentIndex() == 1:
                return unit2.mouseReleaseEvent(self,e)
            if self.ui.tabWidget.currentIndex() == 2:
                return unit3.mouseReleaseEvent(self,e)

    def mousePressEvent(self, e):
        if Qt.LeftButton:
            if self.ui.tabWidget.currentIndex() == 1:
                return unit2.mousePressEvent(self,e)
            if self.ui.tabWidget.currentIndex() == 2:
                return unit3.mousePressEvent(self,e)

    def mouseMoveEvent(self, e):
        if Qt.LeftButton and self.m_drag:
            if self.ui.tabWidget.currentIndex() == 1:
                return unit2.mouseMoveEvent(self,e)
            if self.ui.tabWidget.currentIndex() == 2:
                return unit3.mouseMoveEvent(self,e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    Dlg = MainDialog()
    Dlg.show()
    sys.exit(app.exec_())
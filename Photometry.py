import os, time, sys, threading
from os import listdir
from os.path import abspath, isfile, join
from pathlib import Path
import math
import subprocess
import datetime
import time

import numpy as np
from astropy.io import fits
from astropy import wcs
import astropy.units as u
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.modeling import models, fitting
import PIL.Image as PILimage
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus

from ginga import Bindings
from ginga.misc import log
from ginga.qtw.QtHelp import QtGui, QtCore
from ginga.qtw.ImageViewQt import CanvasView, ScrolledView
from ginga.util import iqcalc
from ginga.util.loader import load_data

class FitsViewer(QtGui.QMainWindow):

    def __init__(self, logger, MainWindow):
        super(FitsViewer, self).__init__()
        self.logger = logger

        self.rawfile = ''

        self.iqcalc = iqcalc.IQCalc(self.logger)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(700, 900)

        # create the ginga viewer and configure it
        fi = CanvasView(self.logger, render='widget')
        fi.enable_autocuts('on')
        fi.set_autocut_params('zscale')
        fi.enable_autozoom('on')
        fi.set_zoom_algorithm('rate')
        fi.set_zoomrate(1.4)
        # fi.set_callback('drag-drop', self.drop_file)
        fi.set_bg(0.2, 0.2, 0.2)
        fi.ui_set_active(True)
        self.fitsimage = fi

        # enable some user interaction
        self.bd = fi.get_bindings()
        self.bd.enable_all(True)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtGui.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(9, 9, 681, 641))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.image_display_hbox = QtGui.QHBoxLayout(self.horizontalLayoutWidget)
        self.image_display_hbox.setContentsMargins(0, 0, 0, 0)
        self.image_display_hbox.setObjectName("image_display_hbox")
        w = fi.get_widget()
        w.setObjectName("w")
        self.image_display_hbox.addWidget(w, stretch=1)
        self.targ_info = QtGui.QLabel(self.centralwidget)
        self.targ_info.setGeometry(QtCore.QRect(150, 720, 251, 31))
        self.targ_info.setObjectName("targ_info")
        self.targ_info.setText("Target Instrument Mag: ")
        self.companion_info = QtGui.QLabel(self.centralwidget)
        self.companion_info.setGeometry(QtCore.QRect(130, 760, 281, 31))
        self.companion_info.setObjectName("companion_info")
        self.companion_info.setText("Companion Instrument Mag: ")
        self.target_mag = QtGui.QLabel(self.centralwidget)
        self.target_mag.setGeometry(QtCore.QRect(180, 800, 221, 31))
        self.target_mag.setObjectName("target_mag")
        self.target_mag.setText("Target Magnitude: ")
        self.image_info = QtGui.QLabel(self.centralwidget)
        self.image_info.setGeometry(QtCore.QRect(20, 660, 171, 31))
        self.image_info.setObjectName("image_info")
        self.image_info.setText("Image: ")
        self.ann_readout = QtGui.QLabel(self.centralwidget)
        self.ann_readout.setGeometry(QtCore.QRect(410, 690, 111, 31))
        self.ann_readout.setObjectName("ann_readout")
        self.ann_readout.setText("Sum: ")
        self.cursorReadout = QtGui.QLabel(self.centralwidget)
        self.cursorReadout.setGeometry(QtCore.QRect(20, 690, 291, 31))
        self.cursorReadout.setObjectName("cursorReadout")
        self.cursorReadout.setText("X:                 Y:                    Value:")
        self.wsettarget = QtGui.QPushButton(self.centralwidget)
        self.wsettarget.setGeometry(QtCore.QRect(10, 720, 101, 28))
        self.wsettarget.setObjectName("wsettarget")
        self.wsettarget.clicked.connect(self.set_target)
        self.wopen = QtGui.QPushButton(self.centralwidget)
        self.wopen.setGeometry(QtCore.QRect(590, 720, 93, 28))
        self.wopen.setObjectName("wopen")
        self.wopen.clicked.connect(self.open_file)
        self.wquit = QtGui.QPushButton(self.centralwidget)
        self.wquit.setGeometry(QtCore.QRect(590, 810, 93, 28))
        self.wquit.setObjectName("wquit")
        self.wquit.clicked.connect(QtGui.QApplication.instance().quit)
        self.wcalculatemag = QtGui.QPushButton(self.centralwidget)
        self.wcalculatemag.setGeometry(QtCore.QRect(10, 800, 101, 41))
        self.wcalculatemag.setObjectName("wcalculatemag")
        self.wcalculatemag.clicked.connect(self.calc_mag)
        self.wopendirectory = QtGui.QPushButton(self.centralwidget)
        self.wopendirectory.setGeometry(QtCore.QRect(590, 760, 93, 41))
        self.wopendirectory.setObjectName("wopendirectory")
        self.wopendirectory.clicked.connect(self.open_directory)
        # self.wsubtractimage = QtGui.QPushButton(self.centralwidget)
        # self.wsubtractimage.setGeometry(QtCore.QRect(590, 760, 93, 41))
        # self.wsubtractimage.setObjectName("wsubtractimage")
        # self.wsubtractimage.clicked.connect(self.subtract_image)
        self.wsetcompanion = QtGui.QPushButton(self.centralwidget)
        self.wsetcompanion.setGeometry(QtCore.QRect(10, 760, 101, 28))
        self.wsetcompanion.setObjectName("wsetcompanion")
        self.wsetcompanion.clicked.connect(self.set_companion)
        self.wcut = QtGui.QComboBox(self.centralwidget)
        self.wcut.setGeometry(QtCore.QRect(300, 660, 73, 22))
        self.wcut.setObjectName("wcut")
        for name in fi.get_autocut_methods():
            self.wcut.addItem(name)
        self.wcut.currentIndexChanged.connect(self.cut_change)
        self.wcolor = QtGui.QComboBox(self.centralwidget)
        self.wcolor.setGeometry(QtCore.QRect(400, 660, 73, 22))
        self.wcolor.setObjectName("wcolor")
        for name in fi.get_color_algorithms():
            self.wcolor.addItem(name)
        self.wcolor.currentIndexChanged.connect(self.color_change)
        self.wpreviousimage = QtGui.QPushButton(self.centralwidget)
        self.wpreviousimage.setGeometry(QtCore.QRect(500, 660, 93, 28))
        self.wpreviousimage.setObjectName("wpreviousimage")
        self.wpreviousimage.clicked.connect(self.previous_image)
        self.wpreviousimage.setEnabled(False)
        self.wnextimage = QtGui.QPushButton(self.centralwidget)
        self.wnextimage.setGeometry(QtCore.QRect(600, 660, 93, 28))
        self.wnextimage.setObjectName("wnextimage")
        self.wnextimage.clicked.connect(self.next_image)
        self.wnextimage.setEnabled(False)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        fi.set_callback('cursor-changed', self.motion_cb)
        fi.add_callback('cursor-down', self.btndown)
        self.anndc, self.circdc = self.add_canvas()
        self.anntag = "annulus-tag"
        self.circtag = "circle-tag"
        self.anntargtag = "target-annulus-tag"
        self.circtargtag = "target-circle-tag"
        self.anncomptag = "companion-annulus-tag"
        self.circcomptag = "companion-circle-tag"

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.targ_info.setText(_translate("MainWindow", "Target Instrument Mag:"))
        self.companion_info.setText(_translate("MainWindow", "Companion Instrument Mag:"))
        self.target_mag.setText(_translate("MainWindow", "Target Magnitude:"))
        self.image_info.setText(_translate("MainWindow", "Image: "))
        self.ann_readout.setText(_translate("MainWindow", "Sum: "))
        self.cursorReadout.setText(_translate("MainWindow", "X:                 Y:                    Value:"))
        self.wsettarget.setText(_translate("MainWindow", "Set Target"))
        self.wopen.setText(_translate("MainWindow", "Open File"))
        self.wquit.setText(_translate("MainWindow", "Quit"))
        self.wpreviousimage.setText(_translate("MainWindow", "Previous Image"))
        self.wnextimage.setText(_translate("MainWindow", "Next Image"))
        self.wcalculatemag.setText(_translate("MainWindow", "Calculate\n"
"Magnitude"))
        self.wopendirectory.setText(_translate("MainWindow", "Open\n"
"Directory"))
#         self.wsubtractimage.setText(_translate("MainWindow", "Subtract\n"
# "Image"))
        self.wsetcompanion.setText(_translate("MainWindow", "Set Companion"))



    def add_canvas(self, tag=None):
        # add a canvas to the view
        my_canvas = self.fitsimage.get_canvas()
        AnnCanvas = my_canvas.get_draw_class('annulus')
        CircCanvas = my_canvas.get_draw_class('circle')
        # RecCanvas = my_canvas.get_draw_class('rectangle')
        # CompCanvas = my_canvas.get_draw_class('compass')
        return AnnCanvas, CircCanvas

    def load_file(self, filepath):
        self.rawfile = filepath
        image = load_data(filepath, logger=self.logger)
        self.fitsimage.set_image(image)
        # self.setWindowTitle(filepath)
        width, height = image.get_size()
        data_x, data_y = width / 2.0, height / 2.0
        # x, y = self.fitsimage.get_canvas_xy(data_x, data_y)
        radius = float(max(width, height)) / 20
        # self.fitsimage.get_canvas().add(self.compdc(data_x, data_y, radius, color='skyblue',
        #                                fontsize=8))
        self.bd._orient(self.fitsimage, righthand=False, msg=True)

    def open_file(self):
        res = QtGui.QFileDialog.getOpenFileName(self, "Open FITS file",
                                                '')

        if isinstance(res, tuple):
            fileName = res[0]
        else:
            fileName = str(res)
        if len(fileName) != 0:
            self.load_file(fileName)
            fn = os.path.basename(fileName)
            text = f"Image: {fn}"
            self.image_info.setText(text)

    def open_directory(self):
        res = QtGui.QFileDialog.getExistingDirectory(self, "Select Directory")
        files = os.listdir(res)
        file_list = []
        for file in files:
            file_list.append(res + '/' + file)
        self.file_list = file_list
        self.image_index = 0
        self.max_index = len(self.file_list) - 1
        self.load_file(self.file_list[self.image_index])
        text = f"Image: {self.file_list[self.image_index]}"
        self.image_info.setText(text)
        self.wnextimage.setEnabled(True)

    def next_image(self):
        self.image_index += 1
        self.load_file(self.file_list[self.image_index])
        text = f"Image: {self.file_list[self.image_index]}"
        self.image_info.setText(text)
        self.wpreviousimage.setEnabled(True)
        if self.image_index == self.max_index:
            self.wnextimage.setEnabled(False)

    def previous_image(self):
        self.image_index -= 1
        self.load_file(self.file_list[self.image_index])
        text = f"Image: {self.file_list[self.image_index]}"
        self.image_info.setText(text)
        self.wnextimage.setEnabled(True)
        if self.image_index == 0:
            self.wpreviousimage.setEnabled(False)


    def cut_change(self):
        self.fitsimage.set_autocut_params(self.wcut.currentText())

    def color_change(self):
        self.fitsimage.set_color_algorithm(self.wcolor.currentText())

    def motion_cb(self, viewer, button, data_x, data_y):

        # Get the value under the data coordinates
        try:
            # We report the value across the pixel, even though the coords
            # change halfway across the pixel
            value = viewer.get_data(int(data_x + 0.5), int(data_y + 0.5))

        except Exception:
            value = None

        fits_x, fits_y = data_x, data_y

        # Calculate WCS RA
        try:
            # NOTE: image function operates on DATA space coords
            image = viewer.get_image()
            if image is None:
                # No image loaded
                return
            ra_txt, dec_txt = image.pixtoradec(fits_x, fits_y,
                                               format='str', coords='fits')
        except Exception as e:
            self.logger.warning("Bad coordinate conversion: %s" % (
                str(e)))
            ra_txt = 'BAD WCS'
            dec_txt = 'BAD WCS'

        if value == None:
            text = "X: %.2f            Y: %.2f               Value: %s" % (fits_x, fits_y, value)
        else:
            text = "X: %.2f  Y: %.2f  Value: %.2f" % (fits_x, fits_y, float(value))
        self.cursorReadout.setText(text)


    def writeFits(self, headerinfo, image_data):
        hdu = fits.PrimaryHDU(header=headerinfo, data=image_data)
        filename = 'procImage.fits'
        try:
            hdu.writeto(filename)
        except OSError:
            self.close()
            os.remove(filename)
            hdu.writeto(filename)
        return filename

    ##Find star stuff
    def cutdetail(self, image, shape_obj):
        view, mask = image.get_shape_view(shape_obj)

        data = image._slice(view)

        return data

    # def backdetail(self, image, x, y):
    #     obj = self.pickannulus
    #     data = self.cutdetail(image, obj)
    #     circ = self.circdc(x, y, 40, color='red')
    #     circdata = self.cutdetail(image, circ)
    #     ann_data = data - circdata
    #     return ann_data


    def findstar(self, image, shape):
        # obj = self.circdc(self.xclick, self.yclick, 60, color='red')
        # shape = obj
        data = self.cutdetail(image, shape)
        ht, wd = data.shape[:2]
        xc, yc = wd // 2, ht // 2
        radius = min(xc, yc)
        peaks = [(xc, yc)]
        peaks = self.iqcalc.find_bright_peaks(data,
                                              threshold=None,
                                              radius=radius)

        xc, yc = peaks[0]
        xc += 1
        yc += 1
        return int(xc), int(yc), data

    def staraperature(self, xc, yc, image):
        positions = np.transpose((xc, yc))
        aperture = CircularAperture(positions, r=6.)
        annulus_aperture = CircularAnnulus(positions, r_in=12., r_out=15.)
        # back_table = aperture_photometry(background, annulus_aperture)
        # print(back_table)
        apers = [aperture, annulus_aperture]
        phot_table = aperture_photometry(image, apers)
        bkg_mean = phot_table['aperture_sum_1'] / annulus_aperture.area
        bkg_sum = bkg_mean * aperture.area
        final_sum = phot_table['aperture_sum_0'] - bkg_sum
        phot_table['residual_aperture_sum'] = final_sum
        for col in phot_table.colnames:
            phot_table[col].info.format = '%.8g'  # for consistent table output
        return phot_table[0]["residual_aperture_sum"]

    def pickstar(self):
        try:
            self.fitsimage.get_canvas().get_object_by_tag(self.anntag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.anntag)
            self.fitsimage.get_canvas().get_object_by_tag(self.circtag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.circtag)
            self.pickcircle = self.circdc(self.xclick, self.yclick, 6, color='red')
            self.pickannulus = self.anndc(self.xclick, self.yclick, 12, 3)
            self.fitsimage.get_canvas().add(self.pickcircle, tag=self.circtag, redraw=True)
            self.fitsimage.get_canvas().add(self.pickannulus, tag=self.anntag, redraw=True)
        except KeyError:
            self.pickcircle = self.circdc(self.xclick, self.yclick, 6, color='red')
            self.pickannulus = self.anndc(self.xclick, self.yclick, 12, 3)
            self.fitsimage.get_canvas().add(self.pickcircle, tag=self.circtag, redraw=True)
            self.fitsimage.get_canvas().add(self.pickannulus, tag=self.anntag, redraw=True)
        image = self.fitsimage.get_image()
        try:
            # data_shape = self.circdc(self.xclick, self.yclick, 60, color='red')
            xc, yc, data = self.findstar(image, self.pickcircle)
            # back_data = self.backdetail(image, xc, yc)
            sum = self.staraperature(self.xclick, self.yclick, image.get_data())
            text = f"Sum: {sum:.2f}"
            self.app_sum = sum
            self.ann_readout.setText(text)
        except IndexError:
            text = "Sum: N/A"
            self.ann_readout.setText(text)

    def set_target(self):
        try:
            target_sum = self.app_sum
        except AttributeError:
            return
        self.targ_mag = -2.5*np.log10(target_sum/90.)
        text = f"Target Instrument Mag:  {self.targ_mag:.2f}"
        self.targ_info.setText(text)
        try:
            self.fitsimage.get_canvas().get_object_by_tag(self.anntargtag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.anntargtag)
            self.fitsimage.get_canvas().get_object_by_tag(self.circtargtag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.circtargtag)
            self.picktargcircle = self.circdc(self.xclick, self.yclick, 6, color='blue')
            self.picktargannulus = self.anndc(self.xclick, self.yclick, 12, 3, color='green')
            self.fitsimage.get_canvas().add(self.picktargcircle, tag=self.circtargtag, redraw=True)
            self.fitsimage.get_canvas().add(self.picktargannulus, tag=self.anntargtag, redraw=True)
        except KeyError:
            self.picktargcircle = self.circdc(self.xclick, self.yclick, 6, color='blue')
            self.picktargannulus = self.anndc(self.xclick, self.yclick, 12, 3, color='green')
            self.fitsimage.get_canvas().add(self.picktargcircle, tag=self.circtargtag, redraw=True)
            self.fitsimage.get_canvas().add(self.picktargannulus, tag=self.anntargtag, redraw=True)

    def set_companion(self):
        try:
            companion_sum = self.app_sum
        except AttributeError:
            return
        self.comp_mag = -2.5*np.log10(companion_sum/90.)
        text = f"Companion Instrument Mag: : {self.comp_mag:.2f}"
        self.companion_info.setText(text)
        try:
            self.fitsimage.get_canvas().get_object_by_tag(self.anncomptag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.anncomptag)
            self.fitsimage.get_canvas().get_object_by_tag(self.circcomptag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.circcomptag)
            self.pickcompcircle = self.circdc(self.xclick, self.yclick, 6, color='red')
            self.pickcompannulus = self.anndc(self.xclick, self.yclick, 12, 3, color='green')
            self.fitsimage.get_canvas().add(self.pickcompcircle, tag=self.circcomptag, redraw=True)
            self.fitsimage.get_canvas().add(self.pickcompannulus, tag=self.anncomptag, redraw=True)
        except KeyError:
            self.pickcompcircle = self.circdc(self.xclick, self.yclick, 6, color='red')
            self.pickcompannulus = self.anndc(self.xclick, self.yclick, 12, 3, color='green')
            self.fitsimage.get_canvas().add(self.pickcompcircle, tag=self.circcomptag, redraw=True)
            self.fitsimage.get_canvas().add(self.pickcompannulus, tag=self.anncomptag, redraw=True)

    def calc_mag(self):
        try:
            diff = self.targ_mag-self.comp_mag
        except AttributeError:
            return
        mag = diff + 11.84
        text = f"Target Magnitude: {mag:.2f}"
        self.target_mag.setText(text)

    # def subtract_image(self):
    #     res = QtGui.QFileDialog.getOpenFileName(self, "Open background file",
    #                                             '')
    #     if isinstance(res, tuple):
    #         fileName = res[0]
    #     else:
    #         fileName = str(res)
    #     if len(fileName) != 0:
    #         self.subtract_sky(fileName)
    #
    # def subtract_sky(self, filename):
    #     targetData = fits.getdata(self.rawfile)
    #     header = fits.getheader(self.rawfile)
    #     skyData = fits.getdata(filename)
    #     with_sky = targetData - skyData
    #     self.load_file(self.writeFits(header, with_sky))


    def btndown(self, canvas, event, data_x, data_y):
        # self.fitsimage.set_pan(data_x, data_y)
        self.xclick = data_x
        self.yclick = data_y
        self.pickstar()


def main():
    ##Write dummy file so walkDirectory caches it in the beginning

    app = QtGui.QApplication([])

    # ginga needs a logger.
    # If you don't want to log anything you can create a null logger by
    # using null=True in this call instead of log_stderr=True
    logger = log.get_logger("example1", log_stderr=True, level=40)

    MainWindow = QtGui.QMainWindow()
    w = FitsViewer(logger, MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

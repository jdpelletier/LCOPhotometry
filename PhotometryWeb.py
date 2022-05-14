import os, time, sys, threading
from os import listdir
from os.path import abspath, isfile, join
from pathlib import Path
import math
import subprocess
import datetime
import time
import logging
import string

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
from ginga.web.pgw import Widgets, Viewers
from ginga.util import iqcalc
from ginga.util.loader import load_data

class FitsViewer(object):

    def __init__(self, logger, window):
        self.logger = logger

        self.rawfile = ''

        self.iqcalc = iqcalc.IQCalc(self.logger)
        self.top = window
        self.top.add_callback('close', self.closed)

        self.file_list = []
        self.image_index = 0

        hbox = Widgets.HBox()
        hbox.set_margins(2, 2, 2, 2)
        hbox.set_spacing(1)

        vbox = Widgets.VBox()
        vbox.set_margins(2, 2, 2, 2)
        vbox.set_spacing(1)

        # create the ginga viewer and configure it
        fi = Viewers.CanvasView(logger)
        fi.enable_autocuts('on')
        fi.set_autocut_params('zscale')
        fi.enable_autozoom('on')
        # fi.set_callback('drag-drop', self.drop_file)
        fi.set_bg(0.2, 0.2, 0.2)
        fi.ui_set_active(True)
        self.fitsimage = fi

        # enable some user interaction
        self.bd = fi.get_bindings()
        self.bd.enable_all(True)

        fi.set_desired_size(600, 600)
        w = Viewers.GingaViewerWidget(viewer=fi)
        vbox.add_widget(w, stretch=1)

        hbox_b = Widgets.HBox()
        hbox_b.set_margins(2, 2, 2, 2)
        hbox_b.set_spacing(4)

        self.image_info = Widgets.Label("Image: ")
        hbox_b.add_widget(self.image_info, stretch=1)

        self.wcut = Widgets.ComboBox()
        for name in fi.get_autocut_methods():
            self.wcut.append_text(name)
        self.wcut.add_callback('activated', self.cut_change)
        self.wcut.set_text('zscale')
        hbox_b.add_widget(self.wcut, stretch=1)

        self.wcolor = Widgets.ComboBox()
        for name in fi.get_color_algorithms():
            self.wcolor.append_text(name)
        self.wcolor.add_callback('activated', self.color_change)
        hbox_b.add_widget(self.wcolor, stretch=1)

        vbox.add_widget(hbox_b, stretch=0)

        hbox_b = Widgets.HBox()
        hbox_b.set_margins(2, 2, 2, 2)
        hbox_b.set_spacing(4)

        self.cursorReadout = Widgets.Label("X:                 Y:                    Value:")
        hbox_b.add_widget(self.cursorReadout, stretch=1)

        self.ann_readout = Widgets.Label("Sum: ")
        hbox_b.add_widget(self.ann_readout, stretch=1)

        vbox.add_widget(hbox_b, stretch=0)

        hbox.add_widget(vbox, stretch=0)

        vbox = Widgets.VBox()
        vbox.set_margins(2, 2, 2, 2)
        vbox.set_spacing(5)

        inst = "Select a directory."

        instructions = Widgets.Label(inst)
        vbox.add_widget(instructions)

        wblue = Widgets.RadioButton("Blue")
        wblue.add_callback('activated', lambda w, val: self.set_directory('Blue', val))
        vbox.add_widget(wblue)

        wgreen = Widgets.RadioButton("Green", group=wblue)
        wgreen.add_callback('activated', lambda w, val: self.set_directory('Green', val))
        vbox.add_widget(wgreen)

        wred = Widgets.RadioButton("Red", group=wblue)
        wred.add_callback('activated', lambda w, val: self.set_directory('Red', val))
        vbox.add_widget(wred)

        inst = "Images:"

        instructions = Widgets.Label(inst)
        vbox.add_widget(instructions)

        hbox_b = Widgets.HBox()
        hbox_b.set_margins(2, 2, 2, 2)
        hbox_b.set_spacing(4)

        self.wimages = Widgets.ComboBox()
        self.wimages.add_callback('activated', self.select_image)
        hbox_b.add_widget(self.wimages, stretch=1)

        for i in range(5):
            spacer = Widgets.Label("")
            hbox_b.add_widget(spacer, stretch=1)

        vbox.add_widget(hbox_b, stretch=0)

        inst = "To zoom, use + or -."

        instructions = Widgets.Label(inst)
        vbox.add_widget(instructions)

        inst = "To pan, hold ctrl, click and drag."

        instructions = Widgets.Label(inst)
        vbox.add_widget(instructions)


        brk = "__________________________________________________________________________"
        brk_label = Widgets.Label(brk)
        vbox.add_widget(brk_label)

        inst = "Click on the target.  It will automatically center the annulus. Then, click SET TARGET."

        instructions = Widgets.Label(inst)
        vbox.add_widget(instructions)

        hbox_b = Widgets.HBox()
        hbox_b.set_margins(2, 2, 2, 2)
        hbox_b.set_spacing(4)

        self.wsettarget = Widgets.Button("SET TARGET")
        self.wsettarget.add_callback('activated', self.set_target)
        self.wsettarget.set_enabled(False)
        hbox_b.add_widget(self.wsettarget, stretch=1)

        for i in range(5):
            spacer = Widgets.Label("")
            hbox_b.add_widget(spacer, stretch=1)

        vbox.add_widget(hbox_b, stretch=0)

        self.targ_info = Widgets.Label("Target Instrument Magnitude: ")
        vbox.add_widget(self.targ_info, stretch=1)

        vbox.add_widget(brk_label)

        inst = "Click on the companion star.  It will automatically center the annulus. Then, click SET COMPANION."

        instructions = Widgets.Label(inst)
        vbox.add_widget(instructions)

        inst = "NOTE: You must select the correct companion star.  See finder chart."

        instructions = Widgets.Label(inst)
        vbox.add_widget(instructions)

        hbox_b = Widgets.HBox()
        hbox_b.set_margins(2, 2, 2, 2)
        hbox_b.set_spacing(4)

        self.wsetcompanion = Widgets.Button("SET COMPANION")
        self.wsetcompanion.add_callback('activated', self.set_companion)
        self.wsetcompanion.set_enabled(False)
        hbox_b.add_widget(self.wsetcompanion, stretch=1)

        for i in range(7):
            spacer = Widgets.Label("")
            hbox_b.add_widget(spacer, stretch=1)

        vbox.add_widget(hbox_b, stretch=0)

        self.companion_info = Widgets.Label("Companion Instrument Magnitude: ")
        vbox.add_widget(self.companion_info, stretch=1)

        vbox.add_widget(brk_label)

        inst = "Once you have set the target and the compantion, click CALCULATE MAGNITUDE."

        instructions = Widgets.Label(inst)
        vbox.add_widget(instructions)

        hbox_b = Widgets.HBox()
        hbox_b.set_margins(2, 2, 2, 2)
        hbox_b.set_spacing(4)

        self.wcalculatemag = Widgets.Button("CALCULATE\n"
"MAGNITUDE")
        self.wcalculatemag.add_callback('activated', self.calc_mag)
        hbox_b.add_widget(self.wcalculatemag, stretch=1)

        for i in range(5):
            spacer = Widgets.Label("")
            hbox_b.add_widget(spacer, stretch=1)

        vbox.add_widget(hbox_b, stretch=0)

        self.file_date = Widgets.Label('Modified Julian Date')
        vbox.add_widget(self.file_date, stretch=1)

        self.target_mag = Widgets.Label('Target Magnitude: ')
        vbox.add_widget(self.target_mag, stretch=1)

        hbox_b = Widgets.HBox()
        hbox_b.set_margins(2, 2, 2, 2)
        hbox_b.set_spacing(4)

        self.wpreviousimage = Widgets.Button("< Previous Image")
        self.wpreviousimage.add_callback('activated', self.previous_image)
        self.wpreviousimage.set_enabled(False)
        hbox_b.add_widget(self.wpreviousimage, stretch=1)

        self.wnextimage = Widgets.Button("Next Image >")
        self.wnextimage.add_callback('activated', self.next_image)
        self.wnextimage.set_enabled(False)
        hbox_b.add_widget(self.wnextimage, stretch=1)

        for i in range(5):
            spacer = Widgets.Label("")
            hbox_b.add_widget(spacer, stretch=1)

        vbox.add_widget(hbox_b, stretch=0)

        hbox.add_widget(vbox, stretch=1)

        vbox = Widgets.VBox()
        vbox.set_margins(2, 2, 2, 2)
        vbox.set_spacing(1)

        #Spacer Zone

        spacer = Widgets.Label("")
        vbox.add_widget(spacer, stretch=1)

        hbox.add_widget(vbox, stretch=1)

        spacer = Widgets.Label("")
        vbox.add_widget(spacer, stretch=1)

        hbox.add_widget(vbox, stretch=1)

        spacer = Widgets.Label("")
        vbox.add_widget(spacer, stretch=1)

        hbox.add_widget(vbox, stretch=1)

        spacer = Widgets.Label("")
        vbox.add_widget(spacer, stretch=1)

        hbox.add_widget(vbox, stretch=1)

        spacer = Widgets.Label("")
        vbox.add_widget(spacer, stretch=1)

        hbox.add_widget(vbox, stretch=1)

        #End Spacer Zone
        self.top.set_widget(hbox)

        fi.set_callback('cursor-changed', self.motion_cb)
        fi.add_callback('cursor-down', self.btndown)
        self.anndc, self.circdc = self.add_canvas()
        self.anntag = "annulus-tag"
        self.circtag = "circle-tag"
        self.datashapetag = "data-tag"
        self.anntargtag = "target-annulus-tag"
        self.circtargtag = "target-circle-tag"
        self.anncomptag = "companion-annulus-tag"
        self.circcomptag = "companion-circle-tag"


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

    def set_directory(self, dir, val):
        self.reset_gui()
        res = f'/home/pi/Desktop/Apps/SN2022hrsData/{dir}'
        files = os.listdir(res)
        file_list = []
        for file in files:
            file_list.append(res + '/' + file)
        self.file_list = file_list
        self.image_index = 0
        self.max_index = len(self.file_list) - 1
        self.load_file(self.file_list[self.image_index])
        text = f"Image: {os.path.basename(self.file_list[self.image_index])}"
        self.image_info.set_text(text)
        self.wnextimage.set_enabled(True)
        self.wpreviousimage.set_enabled(False)
        self.wimages.clear()
        for name in self.file_list:
            self.wimages.insert_alpha(f"{os.path.basename(name)}")
        self.wimages.set_text(f"{os.path.basename(self.file_list[self.image_index])}")


    def next_image(self, clicked):
        self.reset_gui()
        self.image_index += 1
        self.load_file(self.file_list[self.image_index])
        text = f"Image: {os.path.basename(self.file_list[self.image_index])}"
        self.image_info.set_text(text)
        self.wpreviousimage.set_enabled(True)
        self.wimages.set_text(f"{os.path.basename(self.file_list[self.image_index])}")
        if self.image_index == self.max_index:
            self.wnextimage.set_enabled(False)

    def previous_image(self, clicked):
        self.reset_gui()
        self.image_index -= 1
        self.load_file(self.file_list[self.image_index])
        text = f"Image: {os.path.basename(self.file_list[self.image_index])}"
        self.image_info.set_text(text)
        self.wnextimage.set_enabled(True)
        self.wimages.set_text(f"{os.path.basename(self.file_list[self.image_index])}")
        if self.image_index == 0:
            self.wpreviousimage.set_enabled(False)

    def select_image(self, activated, idk):
        self.reset_gui()
        self.image_index = idk
        self.load_file(self.file_list[idk])
        text = f"Image: {os.path.basename(self.file_list[self.image_index])}"
        self.image_info.set_text(text)
        self.wnextimage.set_enabled(True)
        if self.image_index == 0:
            self.wpreviousimage.set_enabled(False)
        if self.image_index == self.max_index:
            self.wnextimage.set_enabled(False)

    def reset_gui(self):
        try:
            self.fitsimage.get_canvas().get_object_by_tag(self.circtargtag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.circtargtag)
            self.fitsimage.get_canvas().get_object_by_tag(self.anntargtag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.anntargtag)
        except KeyError:
            pass
        try:
            self.fitsimage.get_canvas().get_object_by_tag(self.circcomptag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.circcomptag)
            self.fitsimage.get_canvas().get_object_by_tag(self.anncomptag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.anncomptag)
        except KeyError:
            pass
        try:
            self.fitsimage.get_canvas().get_object_by_tag(self.anntag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.anntag)
            self.fitsimage.get_canvas().get_object_by_tag(self.circtag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.circtag)
            self.fitsimage.get_canvas().get_object_by_tag(self.datashapetag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.datashapetag)
        except KeyError:
            pass
        self.targ_info.set_text("Target Instrument Magnitude: ")
        self.companion_info.set_text("Companion Instrument Magnitude: ")
        self.file_date.set_text("Modified Julian Date: ")
        self.target_mag.set_text("Target Magnitude: ")
        self.ann_readout.set_text("Sum: ")
        self.wsettarget.set_enabled(False)
        self.wsetcompanion.set_enabled(False)



    def cut_change(self, activated, idx):
        self.fitsimage.set_autocut_params(self.wcut.get_text())

    def color_change(self, one, two):
        self.fitsimage.set_color_algorithm(self.wcolor.get_text())

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
        self.cursorReadout.set_text(text)


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
        start_x = view[1].start
        start_y = view[0].start

        return data, start_x, start_y


    def findstar(self, image, shape):
        data, start_x, start_y = self.cutdetail(image, shape)
        ht, wd = data.shape[:2]
        xc, yc = wd // 2, ht // 2
        radius = min(xc, yc)
        peaks = [(xc, yc)]
        peaks = self.iqcalc.find_bright_peaks(data,
                                              threshold=None,
                                              radius=radius)

        xc, yc = peaks[0]
        xc = xc + 1 + start_x
        yc = yc + 1 + start_y
        return int(xc), int(yc), data

    def staraperature(self, xc, yc, image):
        positions = np.transpose((xc, yc))
        aperture = CircularAperture(positions, r=6.)
        annulus_aperture = CircularAnnulus(positions, r_in=12., r_out=15.)
        apers = [aperture, annulus_aperture]
        phot_table = aperture_photometry(image, apers)
        bkg_mean = phot_table['aperture_sum_1'] / annulus_aperture.area
        bkg_sum = bkg_mean * aperture.area
        final_sum = phot_table['aperture_sum_0'] - bkg_sum
        phot_table['residual_aperture_sum'] = final_sum
        for col in phot_table.colnames:
            phot_table[col].info.format = '%.8g'  # for consistent table output
        return phot_table[0]["residual_aperture_sum"]

    def set_picking(self, x, y):
        try:
            self.fitsimage.get_canvas().get_object_by_tag(self.anntag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.anntag)
            self.fitsimage.get_canvas().get_object_by_tag(self.circtag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.circtag)
            self.fitsimage.get_canvas().get_object_by_tag(self.datashapetag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.datashapetag)
            self.pickcircle = self.circdc(x, y, 6, color='red')
            self.data_shape = self.circdc(x, y, 15)
            self.pickannulus = self.anndc(x, y, 12, 3)
            self.fitsimage.get_canvas().add(self.pickcircle, tag=self.circtag, redraw=True)
            self.fitsimage.get_canvas().add(self.pickannulus, tag=self.anntag, redraw=True)
        except KeyError:
            self.pickcircle = self.circdc(x, y, 6, color='red')
            self.data_shape = self.circdc(x, y, 15)
            self.pickannulus = self.anndc(x, y, 12, 3)
            self.fitsimage.get_canvas().add(self.pickcircle, tag=self.circtag, redraw=True)
            self.fitsimage.get_canvas().add(self.data_shape, tag=self.datashapetag, redraw=True)
            self.fitsimage.get_canvas().add(self.pickannulus, tag=self.anntag, redraw=True)


    def pickstar(self):
        image = self.fitsimage.get_image()
        try:
            xc, yc, data = self.findstar(image, self.data_shape)
            self.xcenter = xc
            self.ycenter = yc
            sum = self.staraperature(xc, yc, image.get_data())

            text = f"Sum: {sum:.2f}"
            self.app_sum = sum
            self.ann_readout.set_text(text)
            self.set_picking(self.xcenter, self.ycenter)
        except IndexError:
            text = "Sum: N/A"
            self.ann_readout.set_text(text)

    def set_target(self, clicked):
        try:
            target_sum = self.app_sum
        except AttributeError:
            return
        exptime = 90.
        if 'B' in self.image_info.get_text():
            exptime = 120.
        self.targ_mag = -2.5*np.log10(target_sum/exptime)
        text = f"Target Instrument Mag: {self.targ_mag:.2f}"
        self.targ_info.set_text(text)
        try:
            self.fitsimage.get_canvas().get_object_by_tag(self.anntargtag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.anntargtag)
            self.fitsimage.get_canvas().get_object_by_tag(self.circtargtag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.circtargtag)
            self.picktargcircle = self.circdc(self.xcenter, self.ycenter, 6, color='blue')
            self.picktargannulus = self.anndc(self.xcenter, self.ycenter, 12, 3, color='blue')
            self.fitsimage.get_canvas().add(self.picktargcircle, tag=self.circtargtag, redraw=True)
            self.fitsimage.get_canvas().add(self.picktargannulus, tag=self.anntargtag, redraw=True)
        except KeyError:
            self.picktargcircle = self.circdc(self.xcenter, self.ycenter, 6, color='blue')
            self.picktargannulus = self.anndc(self.xcenter, self.ycenter, 12, 3, color='blue')
            self.fitsimage.get_canvas().add(self.picktargcircle, tag=self.circtargtag, redraw=True)
            self.fitsimage.get_canvas().add(self.picktargannulus, tag=self.anntargtag, redraw=True)

    def set_companion(self, clicked):
        try:
            companion_sum = self.app_sum
        except AttributeError:
            return
        exptime = 90.
        if 'B' in self.image_info.get_text():
            exptime = 120.
        self.comp_mag = -2.5*np.log10(companion_sum/exptime)
        text = f"Companion Instrument Magnitude: {self.comp_mag:.2f}"
        self.companion_info.set_text(text)
        try:
            self.fitsimage.get_canvas().get_object_by_tag(self.anncomptag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.anncomptag)
            self.fitsimage.get_canvas().get_object_by_tag(self.circcomptag)
            self.fitsimage.get_canvas().delete_object_by_tag(self.circcomptag)
            self.pickcompcircle = self.circdc(self.xcenter, self.ycenter, 6, color='green')
            self.pickcompannulus = self.anndc(self.xcenter, self.ycenter, 12, 3, color='green')
            self.fitsimage.get_canvas().add(self.pickcompcircle, tag=self.circcomptag, redraw=True)
            self.fitsimage.get_canvas().add(self.pickcompannulus, tag=self.anncomptag, redraw=True)
        except KeyError:
            self.pickcompcircle = self.circdc(self.xcenter, self.ycenter, 6, color='green')
            self.pickcompannulus = self.anndc(self.xcenter, self.ycenter, 12, 3, color='green')
            self.fitsimage.get_canvas().add(self.pickcompcircle, tag=self.circcomptag, redraw=True)
            self.fitsimage.get_canvas().add(self.pickcompannulus, tag=self.anncomptag, redraw=True)

    def calc_mag(self, clicked):
        try:
            diff = self.targ_mag-self.comp_mag
        except AttributeError:
            return
        if 'B' in self.image_info.get_text():
            companion_ab = 13.24
        if 'V' in self.image_info.get_text():
            companion_ab = 11.84
        if 'rp' in self.image_info.get_text():
            companion_ab = 11.8
        mag = diff + companion_ab
        self.calc_mag = mag
        name = self.image_info.get_text().replace(".fits.fz", "").replace("Image: ", "").replace("B", "").replace("rp", "").replace("V", "").replace("-", ".")
        text = f"Modified Julian Date:  {name}\n"
        self.file_date.set_text(text)
        text = f"Target Magnitude: {mag:.2f}"
        self.target_mag.set_text(text)
        # self.wcopy.set_enabled(True)

    # def copy_mag(self, clicked):
    #     name = self.image_info.get_text().replace(".fits.fz", "").replace("Image: ", "").replace("B", "").replace("rp", "").replace("V", "").replace("-", ".")
    #     point = f"{name}\t{self.calc_mag}"
    #     pyperclip.copy(point)
        # with open('outfile.txt', 'a') as fp:
        #     fp.write(point)
        #     fp.write('\n')
        # self.wcopy.set_enabled(False)


    def btndown(self, canvas, event, data_x, data_y):
        # self.fitsimage.set_pan(data_x, data_y)
        self.wsettarget.set_enabled(True)
        self.wsetcompanion.set_enabled(True)
        self.set_picking(data_x, data_y)
        self.pickstar()

    def closed(self, w):
        self.logger.info("Top window closed.")
        w.delete()
        self.top = None
        sys.exit()


def main(options, args):

    # ginga needs a logger.
    # If you don't want to log anything you can create a null logger by
    # using null=True in this call instead of log_stderr=True
    logger = log.get_logger("example1", log_stderr=True, level=40)

    app = Widgets.Application(logger=logger,
                              host=options.host, port=options.port)

    window = app.make_window("SN2022hrs Photometry")

    # our own viewer object, customized with methods (see above)
    viewer = FitsViewer(logger, window)

    if options.renderer is not None:
        render_class = render.get_render_class(options.renderer)
        viewer.fitsimage.set_renderer(render_class(viewer.fitsimage))

    window.resize(700, 900)

    if len(args) > 0:
        viewer.load_file(args[0])

    #window.show()
    #window.raise_()

    try:
        app.mainloop()

    except KeyboardInterrupt:
        logger.info("Terminating viewer...")
        window.close()

if __name__ == "__main__":
    from argparse import ArgumentParser

    argprs = ArgumentParser()

    argprs.add_argument("--debug", dest="debug", default=False, action="store_true",
                        help="Enter the pdb debugger on main()")
    argprs.add_argument("--host", dest="host", metavar="HOST",
                        default='192.168.1.182',
                        help="Listen on HOST for connections")
    argprs.add_argument("--log", dest="logfile", metavar="FILE",
                        help="Write logging output to FILE")
    argprs.add_argument("--loglevel", dest="loglevel", metavar="LEVEL",
                        type=int, default=logging.INFO,
                        help="Set logging level to LEVEL")
    argprs.add_argument("--port", dest="port", metavar="PORT",
                        type=int, default=9909,
                        help="Listen on PORT for connections")
    argprs.add_argument("--profile", dest="profile", action="store_true",
                        default=False,
                        help="Run the profiler on main()")
    argprs.add_argument("-r", "--renderer", dest="renderer", metavar="NAME",
                        default=None,
                        help="Choose renderer (pil|agg|opencv|cairo)")
    argprs.add_argument("--stderr", dest="logstderr", default=False,
                        action="store_true",
                        help="Copy logging also to stderr")
    argprs.add_argument("-t", "--toolkit", dest="toolkit", metavar="NAME",
                        default='qt',
                        help="Choose GUI toolkit (gtk|qt)")

    (options, args) = argprs.parse_known_args(sys.argv[1:])

    # Are we debugging this?
    if options.debug:
        import pdb

        pdb.run('main(options, args)')

    # Are we profiling this?
    elif options.profile:
        import profile

        print(("%s profile:" % sys.argv[0]))
        profile.run('main(options, args)')

    else:
        main(options, args)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import traceback

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RangeSlider, Slider

from utils.bezier import *
from utils.cluster_error import (ClusterNotGeneratedError,
                                 OutOfBoundsClusterError, UndoError)
from utils.generator import (generate_cluster, save_generate, undo_func,
                             update_cluster)

from collections import Counter

# ? Configure logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", fmt="%(asctime)s - %(message)s", datefmt="%H:%M:%S")

import matplotlib
matplotlib.use("Qt5Agg")


class TCG(object):

    def __init__(self):

        dim_x = 1280                            #* Horizonal pixel dimension
        dim_y = 720                             #* Vertical pixel dimension
        limits = [-5, 15]                       #* Curve View axis limits
        extent = limits * 2
        aspect_ratio = dim_x / dim_y
        bg_path = "./bg_images/"
        self.bg_list = [bg_path + s for s in os.listdir(bg_path)]

        axis_color = "#ede5c0"
        slider_color = "#bf616a"

        self.fig = plt.figure("TCG", (14, 8))
        self.fig.suptitle(
            "Trash Cluster Generator", fontsize=14, y=0.95
        )
        self.fig.patch.set_facecolor("#D8DEE9")
        self.ax_bez = self.fig.add_subplot(121)
        self.ax_img = self.fig.add_subplot(122)

        self.fig.subplots_adjust(left=0.12, bottom=0.45, top=0.99, right=0.82)

        #? Slider init values
        
        self.rad = 0.5                          #* Radius
        self.edgy = 0.5                         #* Edginess
        self.seeder = 50                        #* Random Seed
        self.c = [-1, 1]                        #* Translation co-ords
        self.scale = 10                         #* Scale
        self.points = 4                         #* Number of control points
        self.cluster_limit = (5, 10)            #* Cluster Limit

        #? Images
        
        self.bg_index = 0                       #* Background image pointer
        self.cluster_image = None               #* Generated Cluster image
        self.cluster_mask = None                #* Segmentation Mask (RGB)
        self.cluster_pil = None                 #* Ground Truth Segmentation Mask
        self.cache = None                       #* Reference params

        #? Random Bezier Control Points     
        self.a = get_random_points(self.seeder, n=self.points, scale=self.scale) + self.c

        #? Bezier Curve coordinates and centre
        self.x, self.y, self.s = get_bezier_curve(self.a, rad=self.rad, edgy=self.edgy)
        self.centre = np.array([(np.max(self.x) + np.min(self.x)) / 2, (np.max(self.y) + np.min(self.y)) / 2])
        self.a_new = np.append(self.a, [self.centre], axis=0)

        self.ax_bez.set_xlim(limits)
        self.ax_bez.set_ylim(limits)

        #? Curve View Image handler
        self.bezier_handler = self.ax_bez.imshow(
            plt.imread(self.bg_list[self.bg_index]), extent=extent, interpolation="none"
        )

        self.ax_bez.set_aspect(1 / (aspect_ratio))
        self.ax_img.set_aspect(1 / (aspect_ratio))

        #? Curve View Plot and Scatter handler
        (self.bezier_curve,) = self.ax_bez.plot(self.x, self.y, linewidth=1, color="w", zorder=1)
        self.scatter_points = self.ax_bez.scatter(
            self.a_new[:, 0], self.a_new[:, 1], color="orangered", marker=".", alpha=1, zorder=2
        )

        #? Generator View Image handler
        self.cluster_handler = self.ax_img.imshow(
            np.flipud(np.array(plt.imread(self.bg_list[0]))), origin="lower", interpolation="none"
        )

        #? State Indicator Text handler
        self.text_handler = plt.figtext(0.88, 0.70, " Ready ", fontsize=14, backgroundcolor="#a3be8c")

        #? Class distribution indicator text handlers
        class_dict = {"G:": (0.86,"#bdae93"), "M:": (0.901,"#81a1c1"), "P:": (0.942,"#b48ead")}
        self.class_handlers = []
        for (cname, textarg) in class_dict.items():
            self.class_handlers.append(plt.figtext(textarg[0], 0.80, cname+"0".rjust(3), fontsize=10, backgroundcolor=textarg[1]))

        #? Saved image count text handler
        count = len([f for f in os.listdir(".") if f.startswith("label_")])
        self.count_handler = plt.figtext(
            0.86, 0.76, f"Generated images: {count}", fontsize=10, backgroundcolor="#cf9f91"
        )

        undo_asset = plt.imread("./assets/undo.png")

        #? Sliders

        rad_slider_ax = self.fig.add_axes([0.12, 0.42, 0.65, 0.03], facecolor=axis_color)
        self.rad_slider = Slider(rad_slider_ax, "Radius", 0.0, 1.0, valinit=self.rad, color=slider_color)

        edgy_slider_ax = self.fig.add_axes([0.12, 0.37, 0.65, 0.03], facecolor=axis_color)
        self.edgy_slider = Slider(
            edgy_slider_ax, "Edginess", 0.0, 5.0, valinit=self.edgy, color=slider_color
        )

        c0_slider_ax = self.fig.add_axes([0.12, 0.32, 0.65, 0.03], facecolor=axis_color)
        self.c0_slider = Slider(c0_slider_ax, "Move X", -7.0, 16.0, valinit=self.c[0], color=slider_color)

        c1_slider_ax = self.fig.add_axes([0.12, 0.27, 0.65, 0.03], facecolor=axis_color)
        self.c1_slider = Slider(c1_slider_ax, "Move Y", -7.0, 16.0, valinit=self.c[1], color=slider_color)

        scale_slider_ax = self.fig.add_axes([0.12, 0.22, 0.65, 0.03], facecolor=axis_color)
        self.scale_slider = Slider(
            scale_slider_ax, "Scale", 1.0, 20.0, valinit=self.scale, color=slider_color
        )

        points_slider_ax = self.fig.add_axes([0.12, 0.17, 0.65, 0.03], facecolor=axis_color)
        self.points_slider = Slider(
            points_slider_ax, "Points", 3, 10, valinit=self.points, valfmt="%d", color=slider_color
        )

        seeder_slider_ax = self.fig.add_axes([0.12, 0.12, 0.65, 0.03], facecolor=axis_color)
        self.seeder_slider = Slider(
            seeder_slider_ax, "Seed", 1, 100, valinit=self.seeder, valfmt="%d", color=slider_color
        )

        cluster_limit_slider_ax = self.fig.add_axes([0.12, 0.07, 0.65, 0.03], facecolor=axis_color)
        self.cluster_limit_slider = RangeSlider(
            cluster_limit_slider_ax,
            "Cluster Count",
            1,
            20,
            valinit=self.cluster_limit,
            valfmt="%d",
            color=slider_color,
        )

        #? Buttons

        save_button_ax = self.fig.add_axes([0.85, 0.05, 0.1, 0.06])
        self.save_button = Button(save_button_ax, "Save", color="#aee3f2", hovercolor="#85cade")

        reset_button_ax = self.fig.add_axes([0.85, 0.12, 0.1, 0.06])
        self.reset_button = Button(reset_button_ax, "Reset", color="#aee3f2", hovercolor="#85cade")

        background_button_ax = self.fig.add_axes([0.85, 0.19, 0.1, 0.06])
        self.background_button = Button(
            background_button_ax, "Background", color="#aee3f2", hovercolor="#85cade"
        )

        generate_button_ax = self.fig.add_axes([0.85, 0.26, 0.1, 0.06])
        self.generate_button = Button(
            generate_button_ax, "Generate", color="#aee3f2", hovercolor="#85cade"
        )

        add_new_button_ax = self.fig.add_axes([0.85, 0.33, 0.1, 0.06])
        self.add_new_button = Button(
            add_new_button_ax, "Add Cluster", color="#aee3f2", hovercolor="#85cade"
        )

        update_button_ax = self.fig.add_axes([0.85, 0.40, 0.1, 0.06])
        self.update_button = Button(
            update_button_ax, "Update", color="#aee3f2", hovercolor="#85cade"
        )

        undo_button_ax = self.fig.add_axes([0.875, 0.50, 0.05, 0.06])
        self.undo_button = Button(
            undo_button_ax, "", image=undo_asset, color="#aee3f2", hovercolor="#85cade"
        )

        #? Slider event triggers
        self.rad_slider.on_changed(self.sliders_on_changed)
        self.edgy_slider.on_changed(self.sliders_on_changed)
        self.c0_slider.on_changed(self.sliders_on_changed)
        self.c1_slider.on_changed(self.sliders_on_changed)
        self.scale_slider.on_changed(self.sliders_on_changed)
        self.points_slider.on_changed(self.sliders_on_changed)
        self.seeder_slider.on_changed(self.sliders_on_changed)
        self.cluster_limit_slider.on_changed(self.sliders_on_changed)

        #? Button event triggers
        # self.save_button.on_clicked(self.save_button_on_clicked)
        # self.save_button.on_clicked(self.sliders_on_changed)
        # self.reset_button.on_clicked(self.reset_button_on_clicked)
        # self.reset_button.on_clicked(self.sliders_on_changed)
        # self.background_button.on_clicked(self.background_button_on_clicked)
        # self.background_button.on_clicked(self.sliders_on_changed)
        # self.generate_button.on_clicked(self.sliders_on_changed)
        # self.generate_button.on_clicked(self.generate_button_on_clicked)
        # self.add_new_button.on_clicked(self.sliders_on_changed)
        # self.add_new_button.on_clicked(self.add_new_button_on_clicked)
        # self.update_button.on_clicked(self.sliders_on_changed)
        # self.update_button.on_clicked(self.update_on_clicked)
        # self.undo_button.on_clicked(self.sliders_on_changed)
        # self.undo_button.on_clicked(self.undo_on_clicked)

        plt.show()

    def sliders_on_changed(self, val):

        self.cluster_limit = (int(self.cluster_limit_slider.val[0]), int(self.cluster_limit_slider.val[1]))
        self.c = [self.c0_slider.val, self.c1_slider.val]
        self.scale = self.scale_slider.val
        self.a = (
            get_random_points(int(self.seeder_slider.val), n=int(self.points_slider.val), scale=self.scale)
            + self.c
        )
        self.x, self.y, _ = get_bezier_curve(self.a, rad=self.rad_slider.val, edgy=self.edgy_slider.val)
        self.centre = np.array([(np.max(self.x) + np.min(self.x)) / 2, (np.max(self.y) + np.min(self.y)) / 2])
        self.a_new = np.append(self.a, [self.centre], axis=0)

        self.bezier_curve.set_data(self.x, self.y)
        self.scatter_points.set_offsets(self.a_new)
        self.fig.canvas.draw_idle()

  
TCG()





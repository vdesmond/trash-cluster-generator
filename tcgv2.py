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

        dim_x = 1280
        dim_y = 720
        limits = [-5, 15]
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









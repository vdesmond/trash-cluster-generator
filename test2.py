#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RangeSlider
from bezier import *
from generator import *
from cluster_error import ClusterNotGeneratedError, OutOfBoundsClusterError, UndoError

import matplotlib
matplotlib.use('Qt5Agg')

axis_color = "#ede5c0"
slider_color = "#bf616a"

DIM_X = 1280
DIM_Y = 720
LIMITS = [-5, 15]
EXTENT = LIMITS * 2
aspect_ratio = DIM_X / DIM_Y
BG_PATH = "./beach_images/"
BG_LIST = [BG_PATH + s for s in os.listdir(BG_PATH)]

bernstein = lambda n, k, t: binom(n, k) * t ** k * (1.0 - t) ** (n - k)

axis_color = "#ede5c0"
slider_color = "#bf616a"

fig = plt.figure("Bezier Closed Curve Cluster Generator", (14, 8))
fig.suptitle(
    "Interactive Cluster Generator using Random Bezier Closed Curves", fontsize=12
)
fig.patch.set_facecolor("#D8DEE9")
ax_bez = fig.add_subplot(121)
ax_img = fig.add_subplot(122)

fig.subplots_adjust(left=0.15, bottom=0.45, top=0.99, right=0.85)

rad = 0.5
edgy = 0.5
seeder = 50
c = [-1, 1]
scale = 10
points = 4
cluster_limit = (5, 10)

bg_index = 0
cluster_image, cluster_mask, cluster_pil, cache = None, None, None, None

a = get_random_points(seeder, n=points, scale=scale) + c
x, y, s = get_bezier_curve(a, rad=rad, edgy=edgy)
cooordinates = np.array((x, y)).T
centre = np.array([(np.max(x) + np.min(x)) / 2, (np.max(y) + np.min(y)) / 2])
a_new = np.append(a, [centre], axis=0)

bezier_handler = ax_bez.imshow(plt.imread(BG_LIST[bg_index]), extent=EXTENT, interpolation='bicubic')

ax_bez.set_xlim(LIMITS)
ax_bez.set_ylim(LIMITS)

rad_slider_ax = fig.add_axes([0.15, 0.42, 0.65, 0.03], facecolor=axis_color)
rad_slider = Slider(rad_slider_ax, "Radius", 0.0, 1.0, valinit=rad, color=slider_color)

edgy_slider_ax = fig.add_axes([0.15, 0.37, 0.65, 0.03], facecolor=axis_color)
edgy_slider = Slider(
    edgy_slider_ax, "Edginess", 0.0, 5.0, valinit=edgy, color=slider_color
)

c0_slider_ax = fig.add_axes([0.15, 0.32, 0.65, 0.03], facecolor=axis_color)
c0_slider = Slider(c0_slider_ax, "Move X", -7.0, 16.0, valinit=c[0], color=slider_color)

c1_slider_ax = fig.add_axes([0.15, 0.27, 0.65, 0.03], facecolor=axis_color)
c1_slider = Slider(c1_slider_ax, "Move Y", -7.0, 16.0, valinit=c[1], color=slider_color)

scale_slider_ax = fig.add_axes([0.15, 0.22, 0.65, 0.03], facecolor=axis_color)
scale_slider = Slider(
    scale_slider_ax, "Scale", 1.0, 20.0, valinit=scale, color=slider_color
)

points_slider_ax = fig.add_axes([0.15, 0.17, 0.65, 0.03], facecolor=axis_color)
points_slider = Slider(
    points_slider_ax, "Points", 3, 10, valinit=points, valfmt="%d", color=slider_color
)

seeder_slider_ax = fig.add_axes([0.15, 0.12, 0.65, 0.03], facecolor=axis_color)
seeder_slider = Slider(
    seeder_slider_ax, "Seed", 1, 100, valinit=seeder, valfmt="%d", color=slider_color
)

plt.show()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import traceback

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RangeSlider, Slider

from utils.bezier import get_random_points, get_bezier_curve
from utils.cluster_error import (
    ClusterNotGeneratedError,
    OutOfBoundsClusterError,
    UndoError,
)
from utils.generator import generate_cluster, save_generate, undo_func, update_cluster

from collections import Counter

# ? Configure logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", fmt="%(asctime)s - %(message)s", datefmt="%H:%M:%S")

import matplotlib

matplotlib.use("Qt5Agg")


class TCG(object):

    # @profile
    def __init__(self):

        self.dim_x = 1280               # * Horizonal pixel dimension
        self.dim_y = 720                # * Vertical pixel dimension
        self.limits = [-5, 15]          # * Curve View axis limits
        extent = self.limits * 2
        aspect_ratio = self.dim_x / self.dim_y
        bg_path = "./bg_images/"
        self.bg_list = [bg_path + s for s in os.listdir(bg_path)]

        axis_color = "#ede5c0"
        slider_color = "#bf616a"

        self.fig = plt.figure("TCG", (14, 8))
        self.fig.suptitle("Trash Cluster Generator", fontsize=14, y=0.95)
        self.fig.patch.set_facecolor("#D8DEE9")
        self.ax_bez = self.fig.add_subplot(121)
        self.ax_img = self.fig.add_subplot(122)

        self.fig.subplots_adjust(left=0.12, bottom=0.45, top=0.99, right=0.82)

        # ? Slider init values

        self.rad = 0.5                  # * Radius
        self.edgy = 0.5                 # * Edginess
        self.seeder = 50                # * Random Seed
        self.c = [-1, 1]                # * Translation co-ords
        self.scale = 10                 # * Scale
        self.points = 4                 # * Number of control points
        self.cluster_limit = (5, 10)    # * Cluster Limit

        # ? Images

        self.bg_index = 0               # * Background image pointer
        self.cluster_image = None       # * Generated Cluster image
        self.cluster_mask = None        # * Segmentation Mask (RGB)
        self.cluster_pil = None         # * Ground Truth Segmentation Mask
        self.cache = None               # * Reference params

        # ? Random Bezier Control Points
        self.a = (
            get_random_points(self.seeder, n=self.points, scale=self.scale) + self.c
        )

        # ? Bezier Curve coordinates and centre
        self.x, self.y, self.s = get_bezier_curve(self.a, rad=self.rad, edgy=self.edgy)
        self.centre = np.array(
            [
                (np.max(self.x) + np.min(self.x)) / 2,
                (np.max(self.y) + np.min(self.y)) / 2,
            ]
        )
        self.a_new = np.append(self.a, [self.centre], axis=0)

        self.ax_bez.set_xlim(self.limits)
        self.ax_bez.set_ylim(self.limits)

        # ? Curve View Image handler
        self.bezier_handler = self.ax_bez.imshow(
            plt.imread(self.bg_list[self.bg_index]), extent=extent, interpolation="none"
        )

        self.ax_bez.set_aspect(1 / (aspect_ratio))
        self.ax_img.set_aspect(1 / (aspect_ratio))

        # ? Curve View Plot and Scatter handler
        (self.bezier_curve,) = self.ax_bez.plot(
            self.x, self.y, linewidth=1, color="w", zorder=1
        )
        self.scatter_points = self.ax_bez.scatter(
            self.a_new[:, 0],
            self.a_new[:, 1],
            color="orangered",
            marker=".",
            alpha=1,
            zorder=2,
        )

        # ? Generator View Image handler
        self.cluster_handler = self.ax_img.imshow(
            np.flipud(np.array(plt.imread(self.bg_list[0]))),
            origin="lower",
            interpolation="none",
        )

        # ? State Indicator Text handler
        self.text_handler = plt.figtext(
            0.88, 0.70, " Ready ", fontsize=14, backgroundcolor="#a3be8c"
        )

        # ? Class distribution and its indicator text handlers
        self.class_dict = {
            "G:": (0.86, "#bdae93"),
            "M:": (0.901, "#81a1c1"),
            "P:": (0.942, "#b48ead"),
        }
        self.class_handlers = []
        self.class_count = Counter()
        for (cname, textarg) in self.class_dict.items():
            self.class_handlers.append(
                plt.figtext(
                    textarg[0],
                    0.80,
                    cname + "0".rjust(3),
                    fontsize=10,
                    backgroundcolor=textarg[1],
                )
            )

        # ? Saved image count and its indicator text handler
        self.count = len([f for f in os.listdir(".") if f.startswith("label_")])
        self.count_handler = plt.figtext(
            0.86,
            0.76,
            f"Generated images: {self.count}",
            fontsize=10,
            backgroundcolor="#cf9f91",
        )

        undo_asset = plt.imread("./assets/undo.png")

        # ? Sliders

        rad_slider_ax = self.fig.add_axes(
            [0.12, 0.42, 0.65, 0.03], facecolor=axis_color
        )
        self.rad_slider = Slider(
            rad_slider_ax, "Radius", 0.0, 1.0, valinit=self.rad, color=slider_color
        )

        edgy_slider_ax = self.fig.add_axes(
            [0.12, 0.37, 0.65, 0.03], facecolor=axis_color
        )
        self.edgy_slider = Slider(
            edgy_slider_ax, "Edginess", 0.0, 5.0, valinit=self.edgy, color=slider_color
        )

        c0_slider_ax = self.fig.add_axes([0.12, 0.32, 0.65, 0.03], facecolor=axis_color)
        self.c0_slider = Slider(
            c0_slider_ax, "Move X", -7.0, 16.0, valinit=self.c[0], color=slider_color
        )

        c1_slider_ax = self.fig.add_axes([0.12, 0.27, 0.65, 0.03], facecolor=axis_color)
        self.c1_slider = Slider(
            c1_slider_ax, "Move Y", -7.0, 16.0, valinit=self.c[1], color=slider_color
        )

        scale_slider_ax = self.fig.add_axes(
            [0.12, 0.22, 0.65, 0.03], facecolor=axis_color
        )
        self.scale_slider = Slider(
            scale_slider_ax, "Scale", 1.0, 20.0, valinit=self.scale, color=slider_color
        )

        points_slider_ax = self.fig.add_axes(
            [0.12, 0.17, 0.65, 0.03], facecolor=axis_color
        )
        self.points_slider = Slider(
            points_slider_ax,
            "Points",
            3,
            10,
            valinit=self.points,
            valfmt="%d",
            color=slider_color,
        )

        seeder_slider_ax = self.fig.add_axes(
            [0.12, 0.12, 0.65, 0.03], facecolor=axis_color
        )
        self.seeder_slider = Slider(
            seeder_slider_ax,
            "Seed",
            1,
            100,
            valinit=self.seeder,
            valfmt="%d",
            color=slider_color,
        )

        cluster_limit_slider_ax = self.fig.add_axes(
            [0.12, 0.07, 0.65, 0.03], facecolor=axis_color
        )
        self.cluster_limit_slider = RangeSlider(
            cluster_limit_slider_ax,
            "Cluster Count",
            1,
            20,
            valinit=self.cluster_limit,
            valfmt="%d",
            color=slider_color,
        )

        # ? Buttons

        save_button_ax = self.fig.add_axes([0.85, 0.05, 0.1, 0.06])
        self.save_button = Button(
            save_button_ax, "Save", color="#aee3f2", hovercolor="#85cade"
        )

        reset_button_ax = self.fig.add_axes([0.85, 0.12, 0.1, 0.06])
        self.reset_button = Button(
            reset_button_ax, "Reset", color="#aee3f2", hovercolor="#85cade"
        )

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

        # ? Slider event triggers
        self.rad_slider.on_changed(self.curveplot)
        self.edgy_slider.on_changed(self.curveplot)
        self.c0_slider.on_changed(self.curveplot)
        self.c1_slider.on_changed(self.curveplot)
        self.scale_slider.on_changed(self.curveplot)
        self.points_slider.on_changed(self.curveplot)
        self.seeder_slider.on_changed(self.curveplot)
        self.cluster_limit_slider.on_changed(self.curveplot)

        # ? Button event triggers
        self.save_button.on_clicked(self.save)
        self.save_button.on_clicked(self.curveplot)
        self.reset_button.on_clicked(self.reset)
        self.reset_button.on_clicked(self.curveplot)
        self.background_button.on_clicked(self.curveplot)
        self.background_button.on_clicked(self.background)
        self.generate_button.on_clicked(self.curveplot)
        self.generate_button.on_clicked(self.generate)
        self.add_new_button.on_clicked(self.curveplot)
        self.add_new_button.on_clicked(self.add_new)
        self.update_button.on_clicked(self.curveplot)
        self.update_button.on_clicked(self.update)
        self.undo_button.on_clicked(self.curveplot)
        self.undo_button.on_clicked(self.undo)

        plt.show()

    # @profile
    def curveplot(self, val):

        self.cluster_limit = (
            int(self.cluster_limit_slider.val[0]),
            int(self.cluster_limit_slider.val[1]),
        )
        self.c = [self.c0_slider.val, self.c1_slider.val]
        self.scale = self.scale_slider.val
        self.a = (
            get_random_points(
                int(self.seeder_slider.val),
                n=int(self.points_slider.val),
                scale=self.scale,
            )
            + self.c
        )
        self.x, self.y, _ = get_bezier_curve(
            self.a, rad=self.rad_slider.val, edgy=self.edgy_slider.val
        )
        self.centre = np.array(
            [
                (np.max(self.x) + np.min(self.x)) / 2,
                (np.max(self.y) + np.min(self.y)) / 2,
            ]
        )
        self.a_new = np.append(self.a, [self.centre], axis=0)

        self.bezier_curve.set_data(self.x, self.y)
        self.scatter_points.set_offsets(self.a_new)
        self.fig.canvas.draw_idle()

    # @profile
    def save(self, mouse_event):
        try:

            if self.cluster_image is None:
                raise ClusterNotGeneratedError

            save_generate(self.cluster_image, self.cluster_mask, self.cluster_pil)
            self.cluster_image = None
            self.cluster_mask = None
            self.cluster_pil = None
            self.cache = None

        except ClusterNotGeneratedError:
            logger.warning("Generate cluster before saving.")
            self.text_handler.set_text("Generate new")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#ede5c0")
        except Exception:
            logger.error(traceback.print_exc())
            self.text_handler.set_text("Error. See logs")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#bf616a")
        else:
            logger.debug("Saved successfully.")
            self.text_handler.set_text("Saved")
            self.text_handler.set_position((0.89, 0.70))
            self.text_handler.set_backgroundcolor("#a3be8c")
            self.count += 1
            self.count_handler.set_text(f"Generated images: {self.count}")

    # @profile
    def reset(self, mouse_event):
        try:
            self.rad_slider.reset()
            self.edgy_slider.reset()
            self.c0_slider.reset()
            self.c1_slider.reset()
            self.scale_slider.reset()
            self.points_slider.reset()
            self.seeder_slider.reset()
            self.cluster_limit_slider.set_val((3, 7))
            self.bg_index = 0

            self.cluster_image = None
            self.cluster_mask = None
            self.cluster_pil = None

            self.cluster_handler.set_data(
                np.flipud(np.array(plt.imread(self.bg_list[0])))
            )
            self.bezier_handler.set_data(plt.imread(self.bg_list[self.bg_index]))

        except Exception:
            logger.error(traceback.print_exc())
            self.text_handler.set_text("Error. See logs")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#bf616a")
        else:
            logger.info("Reset state.")
            self.text_handler.set_text("Reset")
            self.text_handler.set_position((0.89, 0.70))
            self.text_handler.set_backgroundcolor("#a3be8c")
            for ind, cname in enumerate(self.class_dict):
                self.class_handlers[ind].set_text(cname + "0".rjust(3))

    # @profile
    def background(self, mouse_event):
        try:
            self.bg_index = (self.bg_index + 1) % len(self.bg_list)
            self.bezier_handler.set_data(plt.imread((self.bg_list[self.bg_index])))
        except Exception:
            logger.error(traceback.print_exc())
            self.text_handler.set_text("Error. See logs")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#bf616a")
        else:
            logger.info("Changed background.")
            self.text_handler.set_text("Changed")
            self.text_handler.set_position((0.88, 0.70))
            self.text_handler.set_backgroundcolor("#a3be8c")

    # @profile
    def generate(self, mouse_event):
        try:
            bg_image = np.array(plt.imread(self.bg_list[self.bg_index]))
            bg_mask = np.array(
                plt.imread(
                    self.bg_list[self.bg_index]
                    .replace("images", "labels")
                    .replace("jpeg", "png")
                )
            )
            params = list(zip(self.x, self.y))
            params.append(tuple(self.centre))
            (
                self.cluster_image,
                self.cluster_mask,
                self.cluster_pil,
                self.cache,
            ) = generate_cluster(
                bg_image,
                bg_mask,
                params,
                self.cluster_limit,
                self.limits,
                (self.dim_x, self.dim_y),
            )

            if not self.cluster_image:
                raise OutOfBoundsClusterError

            self.cluster_handler.set_data(np.flipud(self.cluster_image))

        except OutOfBoundsClusterError:
            logger.warning("Out of Bounds. Retry")
            self.text_handler.set_text("Out of Bounds")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#ede5c0")

        except Exception:
            logger.error(traceback.print_exc())
            self.text_handler.set_text("Error. See logs")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#bf616a")

        else:
            logger.info("Generated new cluster.")
            self.text_handler.set_text("Generated")
            self.text_handler.set_position((0.88, 0.70))
            self.text_handler.set_backgroundcolor("#a3be8c")

            self.class_count = Counter(self.cache[2])
            for ind, cname in enumerate(self.class_dict):
                self.class_handlers[ind].set_text(
                    cname + str(self.class_count[ind + 3]).rjust(3)
                )

    # @profile
    def add_new(self, mouse_event):
        try:

            params = list(zip(self.x, self.y))
            params.append(tuple(self.centre))

            if self.cluster_image is None:
                raise ClusterNotGeneratedError

            (
                self.cluster_image,
                self.cluster_mask,
                self.cluster_pil,
                self.cache,
            ) = generate_cluster(
                np.array(self.cluster_image),
                self.cluster_mask,
                params,
                self.cluster_limit,
                self.limits,
                (self.dim_x, self.dim_y),
                new_cluster=False,
            )

            if np.array_equal(self.cluster_image, self.cache[0]):
                raise OutOfBoundsClusterError

            self.cluster_handler.set_data(np.flipud(self.cluster_image))

        except ClusterNotGeneratedError:
            logger.warning("Generate cluster before adding a new one.")
            self.text_handler.set_text("Generate new")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#ede5c0")

        except OutOfBoundsClusterError:
            logger.warning("Out of Bounds. Retry")
            self.text_handler.set_text("Out of Bounds")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#ede5c0")

        except Exception:
            logger.error(traceback.print_exc())
            self.text_handler.set_text("Error. See logs.")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#bf616a")

        else:
            logger.info("Added cluster.")
            self.text_handler.set_text("Added")
            self.text_handler.set_position((0.89, 0.70))
            self.text_handler.set_backgroundcolor("#a3be8c")

            self.class_count += Counter(self.cache[2])
            for ind, cname in enumerate(self.class_dict):
                self.class_handlers[ind].set_text(
                    cname + str(self.class_count[ind + 3]).rjust(3)
                )

    # @profile
    def update(self, mouse_event):
        try:
            params = list(zip(self.x, self.y))
            params.append(tuple(self.centre))

            if self.cluster_image is None:
                raise ClusterNotGeneratedError

            bg_mask = np.array(
                plt.imread(
                    self.bg_list[self.bg_index]
                    .replace("images", "labels")
                    .replace("jpeg", "png")
                )
            )

            if np.array_equal(bg_mask, self.cache[1]):
                new_cluster = True
            else:
                new_cluster = False

            # old_class_list = cache[2][:]

            (
                self.cluster_image,
                self.cluster_mask,
                self.cluster_pil,
                self.cache,
            ) = update_cluster(
                *self.cache, params, self.limits, (self.dim_x, self.dim_y), new_cluster
            )

            if np.array_equal(self.cluster_image, self.cache[0]):
                raise OutOfBoundsClusterError

            self.cluster_handler.set_data(np.flipud(self.cluster_image))

        except ClusterNotGeneratedError:
            logger.warning("Generate cluster before updating.")
            self.text_handler.set_text("Generate new")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#ede5c0")

        except OutOfBoundsClusterError:
            logger.warning("Out of Bounds. Retry")
            self.text_handler.set_text("Out of Bounds")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#ede5c0")

        except Exception:
            logger.error(traceback.print_exc())
            self.text_handler.set_text("Error. See logs")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#bf616a")

        else:
            logger.info("Updated cluster.")
            self.text_handler.set_text("Updated")
            self.text_handler.set_position((0.88, 0.70))
            self.text_handler.set_backgroundcolor("#a3be8c")

    # @profile
    def undo(self, mouse_event):
        try:

            if self.cluster_image is None:
                raise UndoError

            (
                self.cluster_image,
                self.cluster_mask,
                self.cluster_pil,
                self.cache,
            ) = undo_func()

            if self.cache is None:
                raise UndoError

            self.cluster_handler.set_data(np.flipud(self.cluster_image))

        except UndoError:
            logger.warning("Cannot undo as there is no previous state.")
            self.text_handler.set_text("Cannot undo")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#ede5c0")

        except Exception:
            logger.error(traceback.print_exc())
            self.text_handler.set_text("Error. See logs")
            self.text_handler.set_position((0.87, 0.70))
            self.text_handler.set_backgroundcolor("#bf616a")

        else:
            logger.info("Cluster undone.")
            self.text_handler.set_text("Undone")
            self.text_handler.set_position((0.88, 0.70))
            self.text_handler.set_backgroundcolor("#a3be8c")

            self.class_count -= Counter(self.cache[2])
            for ind, cname in enumerate(self.class_dict):
                self.class_handlers[ind].set_text(
                    cname + str(self.class_count[ind + 3]).rjust(3)
                )


if __name__ == "__main__":
    TCG()

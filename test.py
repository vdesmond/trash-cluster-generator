import os
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RangeSlider
from bezier import *
from generator import *
from cluster_error import ClusterNotGeneratedError, OutOfBoundsClusterError, UndoError

# from memory_profiler import profile
import logging
import coloredlogs

# ? Configure logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", fmt="%(asctime)s - %(message)s", datefmt="%H:%M:%S")

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
centre = np.array([(np.max(x) + np.min(x)) / 2, (np.max(y) + np.min(y)) / 2])
a_new = np.append(a, [centre], axis=0)

ax_bez.set_xlim(LIMITS)
ax_bez.set_ylim(LIMITS)

bezier_handler = ax_bez.imshow(
    plt.imread(BG_LIST[bg_index]), extent=EXTENT
)

ax_bez.set_aspect(1 / (aspect_ratio))
ax_img.set_aspect(1 / (aspect_ratio))

(bezier_curve,) = ax_bez.plot(x, y, linewidth=1, color="w")
scatter_points = ax_bez.scatter(
    a_new[:, 0],
    a_new[:, 1],
    color="r",
    marker=".",
    alpha=1,
)

cluster_handler = ax_img.imshow(
    np.flipud(np.array(plt.imread(BG_LIST[0]))), origin="lower"
)
text_handler = plt.figtext(0.89, 0.70, "Ready", fontsize=14, backgroundcolor="#a3be8c")

count = len([f for f in os.listdir(".") if f.startswith("label_")])
count_handler = plt.figtext(
    0.865, 0.80, f"Generated images: {count}", fontsize=10, backgroundcolor="#cf9f91"
)

undo_asset = plt.imread("./assets/undo.png")

# Sliders
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

cluster_limit_slider_ax = fig.add_axes([0.15, 0.07, 0.65, 0.03], facecolor=axis_color)
cluster_limit_slider = RangeSlider(
    cluster_limit_slider_ax,
    "Cluster Count",
    1,
    20,
    valinit=cluster_limit,
    valfmt="%d",
    color=slider_color,
)

# @profile
def sliders_on_changed(val):
    global cluster_limit
    cluster_limit = (int(cluster_limit_slider.val[0]), int(cluster_limit_slider.val[1]))
    global x, y
    c = [c0_slider.val, c1_slider.val]
    scale = scale_slider.val
    a = (
        get_random_points(int(seeder_slider.val), n=int(points_slider.val), scale=scale)
        + c
    )
    x, y, _ = get_bezier_curve(a, rad=rad_slider.val, edgy=edgy_slider.val)
    global centre
    centre = np.array([(np.max(x) + np.min(x)) / 2, (np.max(y) + np.min(y)) / 2])
    global a_new
    a_new = np.append(a, [centre], axis=0)

    bezier_handler.set_data(plt.imread(BG_LIST[bg_index]))
    bezier_curve.set_data(x, y)
    scatter_points.set_offsets(a_new)
    fig.canvas.draw_idle()


rad_slider.on_changed(sliders_on_changed)
edgy_slider.on_changed(sliders_on_changed)
c0_slider.on_changed(sliders_on_changed)
c1_slider.on_changed(sliders_on_changed)
scale_slider.on_changed(sliders_on_changed)
points_slider.on_changed(sliders_on_changed)
seeder_slider.on_changed(sliders_on_changed)
cluster_limit_slider.on_changed(sliders_on_changed)

# -------------------------------------------------------------------- #
save_button_ax = fig.add_axes([0.85, 0.05, 0.1, 0.06])
save_button = Button(save_button_ax, "Save", color="#aee3f2", hovercolor="#85cade")

# @profile
def save_button_on_clicked(mouse_event):
    try:
        global cluster_image, cluster_mask, cluster_pil, cache
        save_generate(cluster_image, cluster_mask, cluster_pil)
        cluster_image, cluster_mask, cluster_pil, cache = None, None, None, None
    except ClusterNotGeneratedError:
        logger.warning("Generate cluster before saving.")
        text_handler.set_text("Generate new")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#ede5c0")
    except Exception:
        logger.error(traceback.print_exc())
        text_handler.set_text("Error. See logs")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#bf616a")
    else:
        logger.debug("Saved successfully.")
        text_handler.set_text("Saved")
        text_handler.set_position((0.89, 0.70))
        text_handler.set_backgroundcolor("#a3be8c")
        global count
        count += 1
        count_handler.set_text(f"Generated images: {count}")


save_button.on_clicked(save_button_on_clicked)
save_button.on_clicked(sliders_on_changed)

# -------------------------------------------------------------------- #
reset_button_ax = fig.add_axes([0.85, 0.12, 0.1, 0.06])
reset_button = Button(reset_button_ax, "Reset", color="#aee3f2", hovercolor="#85cade")

# @profile
def reset_button_on_clicked(mouse_event):
    try:
        global bg_index
        rad_slider.reset()
        edgy_slider.reset()
        c0_slider.reset()
        c1_slider.reset()
        scale_slider.reset()
        points_slider.reset()
        seeder_slider.reset()
        cluster_limit_slider.set_val((3, 7))
        bg_index = 0

        global cluster_image, cluster_mask, cluster_pil
        cluster_image, cluster_mask, cluster_pil = None, None, None
        cluster_handler.set_data(np.flipud(np.array(plt.imread(BG_LIST[0]))))
    except Exception:
        logger.error(traceback.print_exc())
        text_handler.set_text("Error. See logs")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#bf616a")
    else:
        logger.info("Reset state.")
        text_handler.set_text("Reset")
        text_handler.set_position((0.89, 0.70))
        text_handler.set_backgroundcolor("#a3be8c")


reset_button.on_clicked(reset_button_on_clicked)
reset_button.on_clicked(sliders_on_changed)

# -------------------------------------------------------------------- #

background_button_ax = fig.add_axes([0.85, 0.19, 0.1, 0.06])
background_button = Button(
    background_button_ax, "Background", color="#aee3f2", hovercolor="#85cade"
)

# @profile
def background_button_on_clicked(mouse_event):
    try:
        global bg_index
        bg_index = bg_index + 1 % (len(BG_LIST))
    except Exception:
        logger.error(traceback.print_exc())
        text_handler.set_text("Error. See logs")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#bf616a")
    else:
        logger.info("Changed background.")
        text_handler.set_text("Changed")
        text_handler.set_position((0.88, 0.70))
        text_handler.set_backgroundcolor("#a3be8c")


background_button.on_clicked(background_button_on_clicked)
background_button.on_clicked(sliders_on_changed)

# -------------------------------------------------------------------- #

generate_button_ax = fig.add_axes([0.85, 0.26, 0.1, 0.06])
generate_button = Button(
    generate_button_ax, "Generate", color="#aee3f2", hovercolor="#85cade"
)

# @profile
def generate_button_on_clicked(mouse_event):
    try:
        bg_image = np.array(plt.imread(BG_LIST[bg_index]))
        bg_mask = np.array(
            plt.imread(
                BG_LIST[bg_index].replace("images", "labels").replace("jpeg", "png")
            )
        )
        params = list(zip(x, y))
        params.append(tuple(centre))
        global cluster_image, cluster_mask, cluster_pil, cache
        cluster_image, cluster_mask, cluster_pil, cache = generate_cluster(
            bg_image, bg_mask, params, cluster_limit, LIMITS, (DIM_X, DIM_Y)
        )

        if not cluster_image:
            raise OutOfBoundsClusterError

        cluster_handler.set_data(np.flipud(cluster_image))

    except OutOfBoundsClusterError:
        logger.warning("Out of Bounds. Retry")
        text_handler.set_text("Out of Bounds")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#ede5c0")

    except Exception:
        logger.error(traceback.print_exc())
        text_handler.set_text("Error. See logs")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#bf616a")

    else:
        logger.info("Generated new cluster.")
        text_handler.set_text("Generated")
        text_handler.set_position((0.88, 0.70))
        text_handler.set_backgroundcolor("#a3be8c")


generate_button.on_clicked(sliders_on_changed)
generate_button.on_clicked(generate_button_on_clicked)

# -------------------------------------------------------------------- #

add_new_button_ax = fig.add_axes([0.85, 0.33, 0.1, 0.06])
add_new_button = Button(
    add_new_button_ax, "Add Cluster", color="#aee3f2", hovercolor="#85cade"
)

# @profile
def add_new_button_on_clicked(mouse_event):
    try:

        params = list(zip(x, y))
        params.append(tuple(centre))
        global cluster_image, cluster_mask, cluster_pil, cache

        if cluster_image is None:
            raise ClusterNotGeneratedError

        cluster_image, cluster_mask, cluster_pil, cache = generate_cluster(
            np.array(cluster_image),
            cluster_mask,
            params,
            cluster_limit,
            LIMITS,
            (DIM_X, DIM_Y),
            new_cluster=False,
        )

        if np.array_equal(cluster_image, cache[0]):
            raise OutOfBoundsClusterError

        cluster_handler.set_data(np.flipud(cluster_image))

    except ClusterNotGeneratedError:
        logger.warning("Generate cluster before adding a new one.")
        text_handler.set_text("Generate new")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#ede5c0")

    except OutOfBoundsClusterError:
        logger.warning("Out of Bounds. Retry")
        text_handler.set_text("Out of Bounds")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#ede5c0")

    except Exception:
        logger.error(traceback.print_exc())
        text_handler.set_text("Error. See logs.")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#bf616a")

    else:
        logger.info("Added cluster.")
        text_handler.set_text("Added")
        text_handler.set_position((0.89, 0.70))
        text_handler.set_backgroundcolor("#a3be8c")


add_new_button.on_clicked(sliders_on_changed)
add_new_button.on_clicked(add_new_button_on_clicked)

# -------------------------------------------------------------------- #

update_button_ax = fig.add_axes([0.85, 0.40, 0.1, 0.06])
update_button = Button(
    update_button_ax, "Update", color="#aee3f2", hovercolor="#85cade"
)

# @profile
def update_on_clicked(mouse_event):
    try:

        params = list(zip(x, y))
        params.append(tuple(centre))
        global cluster_image, cluster_mask, cluster_pil, cache

        if cluster_image is None:
            raise ClusterNotGeneratedError

        bg_mask = np.array(
            plt.imread(
                BG_LIST[bg_index].replace("images", "labels").replace("jpeg", "png")
            )
        )

        if np.array_equal(bg_mask, cache[1]):
            new_cluster = True
        else:
            new_cluster = False

        cluster_image, cluster_mask, cluster_pil, cache = update_cluster(
            *cache, params, LIMITS, (DIM_X, DIM_Y), new_cluster
        )

        if np.array_equal(cluster_image, cache[0]):
            raise OutOfBoundsClusterError

        cluster_handler.set_data(np.flipud(cluster_image))

    except ClusterNotGeneratedError:
        logger.warning("Generate cluster before updating.")
        text_handler.set_text("Generate new")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#ede5c0")

    except OutOfBoundsClusterError:
        logger.warning("Out of Bounds. Retry")
        text_handler.set_text("Out of Bounds")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#ede5c0")

    except Exception:
        logger.error(traceback.print_exc())
        text_handler.set_text("Error. See logs")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#bf616a")

    else:
        logger.info("Updated cluster.")
        text_handler.set_text("Updated")
        text_handler.set_position((0.88, 0.70))
        text_handler.set_backgroundcolor("#a3be8c")


update_button.on_clicked(sliders_on_changed)
update_button.on_clicked(update_on_clicked)

# -------------------------------------------------------------------- #

undo_button_ax = fig.add_axes([0.885, 0.60, 0.05, 0.06])
undo_button = Button(
    undo_button_ax, "", image=undo_asset, color="#aee3f2", hovercolor="#85cade"
)

# @profile
def undo_on_clicked(mouse_event):
    try:
        global cluster_image, cluster_mask, cluster_pil, cache

        if cluster_image is None:
            raise UndoError

        cluster_image, cluster_mask, cluster_pil, cache = undo_func()

        if cache is None:
            raise UndoError

        cluster_handler.set_data(np.flipud(cluster_image))

    except UndoError:
        logger.warning("Cannot undo as there is no previous state.")
        text_handler.set_text("Cannot undo")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#ede5c0")

    except Exception:
        logger.error(traceback.print_exc())
        text_handler.set_text("Error. See logs")
        text_handler.set_position((0.87, 0.70))
        text_handler.set_backgroundcolor("#bf616a")

    else:
        logger.info("Cluster undone.")
        text_handler.set_text("Undone")
        text_handler.set_position((0.88, 0.70))
        text_handler.set_backgroundcolor("#a3be8c")


undo_button.on_clicked(sliders_on_changed)
undo_button.on_clicked(undo_on_clicked)

plt.show()

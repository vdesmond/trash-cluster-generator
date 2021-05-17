import os
import traceback
from PIL.Image import new
from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, RangeSlider
from bezier import *
from generator import *
from cluster_error import ClusterNotGeneratedError, OutOfBoundsClusterError

DIM_X = 1280
DIM_Y = 720
LIMITS = [-5, 15]
EXTENT = LIMITS * 2
aspect_ratio = DIM_X / DIM_Y
BG_PATH = "./beach_images/"
BG_LIST = [BG_PATH + s for s in os.listdir(BG_PATH)]

bernstein = lambda n, k, t: binom(n, k) * t ** k * (1.0 - t) ** (n - k)
fill_status = False

axis_color = "#ede5c0"
slider_color = "#BF616A"

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
cluster_limit = (3,7)

bg_index = 0
cluster_image, cluster_mask, cluster_pil, cache = None, None, None, None

a = get_random_points(seeder, n=points, scale=scale) + c
x, y, s = get_bezier_curve(a, rad=rad, edgy=edgy)
cooordinates = np.array((x, y)).T
centre = np.array([(np.max(x) + np.min(x)) / 2, (np.max(y) + np.min(y)) / 2])
a_new = np.append(a, [centre], axis=0)

ax_bez.set_xlim(LIMITS)
ax_bez.set_ylim(LIMITS)
ax_bez.imshow(plt.imread(BG_LIST[bg_index]), extent=EXTENT)
ax_bez.set_aspect(1 / (aspect_ratio))
ax_img.set_aspect(1 / (aspect_ratio))
[line] = ax_bez.plot(x, y, linewidth=1, color="w")
scatter_points = ax_bez.scatter(
    a_new[:, 0],
    a_new[:, 1],
    color="r",
    marker=".",
    alpha=1,
)
ax_img.imshow(np.flipud(np.array(plt.imread(BG_LIST[0]))), origin="lower")

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
    cluster_limit_slider_ax, "Cluster Count", 1, 20, valinit=cluster_limit, valfmt="%d", color=slider_color
)

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
    cooordinates = np.array((x, y)).T
    global centre
    centre = np.array([(np.max(x) + np.min(x)) / 2, (np.max(y) + np.min(y)) / 2])
    global a_new
    a_new = np.append(a, [centre], axis=0)

    global scatter_points
    if not fill_status:
        ax_bez.clear()
        ax_bez.set_xlim(LIMITS)
        ax_bez.set_ylim(LIMITS)
        ax_bez.imshow(plt.imread(BG_LIST[bg_index]), extent=EXTENT)
        ax_bez.set_aspect(1 / (aspect_ratio))
        ax_bez.plot(x, y, linewidth=1, color="w")
    else:
        ax_bez.clear()
        ax_bez.set_xlim(LIMITS)
        ax_bez.set_ylim(LIMITS)
        ax_bez.imshow(plt.imread(BG_LIST[bg_index]), extent=EXTENT)
        ax_bez.set_aspect(1 / (aspect_ratio))
        ax_bez.fill(x, y, color="w", alpha=0.5)
    scatter_points = ax_bez.scatter(
        a_new[:, 0],
        a_new[:, 1],
        color="r",
        marker=".",
        alpha=1,
    )
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


def save_button_on_clicked(mouse_event):
    save_generate(cluster_image, cluster_mask, cluster_pil)


save_button.on_clicked(save_button_on_clicked)
save_button.on_clicked(sliders_on_changed)

# -------------------------------------------------------------------- #
reset_button_ax = fig.add_axes([0.85, 0.12, 0.1, 0.06])
reset_button = Button(reset_button_ax, "Reset", color="#aee3f2", hovercolor="#85cade")


def reset_button_on_clicked(mouse_event):
    global bg_index
    rad_slider.reset()
    edgy_slider.reset()
    c0_slider.reset()
    c1_slider.reset()
    scale_slider.reset()
    points_slider.reset()
    seeder_slider.reset()
    cluster_limit_slider.set_val((3,7))
    bg_index = 0

    global cluster_image, cluster_mask, cluster_pil
    cluster_image, cluster_mask, cluster_pil = None, None, None
    ax_img.imshow(np.flipud(np.array(plt.imread(BG_LIST[0]))), origin="lower")


reset_button.on_clicked(reset_button_on_clicked)
reset_button.on_clicked(sliders_on_changed)

# -------------------------------------------------------------------- #

background_button_ax = fig.add_axes([0.85, 0.19, 0.1, 0.06])
background_button = Button(
    background_button_ax, "Background", color="#aee3f2", hovercolor="#85cade"
)


def background_button_on_clicked(mouse_event):
    global bg_index
    bg_index = bg_index + 1 % (len(BG_LIST))


background_button.on_clicked(background_button_on_clicked)
background_button.on_clicked(sliders_on_changed)

# -------------------------------------------------------------------- #

generate_button_ax = fig.add_axes([0.85, 0.26, 0.1, 0.06])
generate_button = Button(
    generate_button_ax, "Generate", color="#aee3f2", hovercolor="#85cade"
)


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
        cluster_image, cluster_mask, cluster_pil, cache = generate_cluster(bg_image, bg_mask, params, cluster_limit, LIMITS, (DIM_X,DIM_Y))
        
        if not cluster_image:
            raise OutOfBoundsClusterError

        ax_img.imshow(np.flipud(cluster_image), origin="lower")

    except OutOfBoundsClusterError:
        print("Error: Out of Bounds. Retry")
    
    except Exception:
        traceback.print_exc()


generate_button.on_clicked(sliders_on_changed)
generate_button.on_clicked(generate_button_on_clicked)

# -------------------------------------------------------------------- #

add_new_button_ax = fig.add_axes([0.85, 0.33, 0.1, 0.06])
add_new_button = Button(
    add_new_button_ax, "Add Cluster", color="#aee3f2", hovercolor="#85cade"
)


def add_new_button_on_clicked(mouse_event):
    try:
        
        params = list(zip(x, y))
        params.append(tuple(centre))
        global cluster_image, cluster_mask, cluster_pil, cache

        if cluster_image is None:
            raise ClusterNotGeneratedError

        cluster_image, cluster_mask, cluster_pil, cache = generate_cluster(np.array(cluster_image), cluster_mask, params, cluster_limit, LIMITS, (DIM_X,DIM_Y), new_cluster=False)
        
        if cluster_image is None:
            raise OutOfBoundsClusterError

        ax_img.imshow(np.flipud(cluster_image), origin="lower")

    except ClusterNotGeneratedError:
        print("Error: Generate cluster before adding a new one.")

    except OutOfBoundsClusterError:
        print("Error: Out of Bounds. Retry")

    except Exception:
        traceback.print_exc()


add_new_button.on_clicked(sliders_on_changed)
add_new_button.on_clicked(add_new_button_on_clicked)

# -------------------------------------------------------------------- #

update_button_ax = fig.add_axes([0.85, 0.40, 0.1, 0.06])
update_button = Button(
    update_button_ax, "Update", color="#aee3f2", hovercolor="#85cade"
)


def update_on_clicked(mouse_event):
    try:
        
        params = list(zip(x, y))
        params.append(tuple(centre))
        global cluster_image, cluster_mask, cluster_pil, cache

        if cluster_image is None:
            raise ClusterNotGeneratedError

        cluster_image, cluster_mask, cluster_pil, cache = update_cluster(*cache, params, LIMITS, (DIM_X,DIM_Y))
        
        if cluster_image is None:
            raise OutOfBoundsClusterError

        ax_img.imshow(np.flipud(cluster_image), origin="lower")

    except ClusterNotGeneratedError:
        print("Error: Generate cluster before adding a new one.")

    except OutOfBoundsClusterError:
        print("Error: Out of Bounds. Retry")

    except Exception:
        traceback.print_exc()


update_button.on_clicked(sliders_on_changed)
update_button.on_clicked(update_on_clicked)

# -------------------------------------------------------------------- #

fill_radios_ax = fig.add_axes([0.88, 0.5, 0.07, 0.10], facecolor=axis_color)
fill_radios = RadioButtons(fill_radios_ax, ("No fill", "Fill"), active=0)


def fill_radios_on_clicked(label):
    global fill_status
    if label == "Fill":
        fill_status = True
    else:
        fill_status = False


fill_radios.on_clicked(fill_radios_on_clicked)
fill_radios.on_clicked(sliders_on_changed)
# -------------------------------------------------------------------- #

plt.show()
import os
from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from bezier import *
from central_initalize import *

DIM_X = 1280
DIM_Y = 720
LIMITS = [-5, 15]
EXTENT = LIMITS * 2
aspect_ratio = DIM_X / DIM_Y
BG_PATH = "./beach_images/"
BG_LIST = [BG_PATH + s for s in os.listdir(BG_PATH)]

bernstein = lambda n, k, t: binom(n, k) * t ** k * (1.0 - t) ** (n - k)
fill_status = False

axis_color = "lightgoldenrodyellow"

fig = plt.figure("Bezier Closed Curve Generator", (14, 8))
fig.suptitle(
    "Interactive Cluster Generator using Random Bezier Closed Curves", fontsize=12
)
ax_bez = fig.add_subplot(121)
ax_img = fig.add_subplot(122)

fig.subplots_adjust(left=0.1, bottom=0.42, top=0.93, right=0.85)

rad = 0.5
edgy = 0.5
seeder = 50
c = [-1, 1]
scale = 10
points = 4

bg_index = 0

a = get_random_points(seeder, n=points, scale=scale) + c
x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
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

# Sliders
rad_slider_ax = fig.add_axes([0.15, 0.32, 0.65, 0.03], facecolor=axis_color)
rad_slider = Slider(rad_slider_ax, "Radius", 0.0, 1.0, valinit=rad)

edgy_slider_ax = fig.add_axes([0.15, 0.27, 0.65, 0.03], facecolor=axis_color)
edgy_slider = Slider(edgy_slider_ax, "Edginess", 0.0, 5.0, valinit=edgy)

c0_slider_ax = fig.add_axes([0.15, 0.22, 0.65, 0.03], facecolor=axis_color)
c0_slider = Slider(c0_slider_ax, "Move X", -5.0, 5.0, valinit=c[0])

c1_slider_ax = fig.add_axes([0.15, 0.17, 0.65, 0.03], facecolor=axis_color)
c1_slider = Slider(c1_slider_ax, "Move Y", -5.0, 5.0, valinit=c[1])

scale_slider_ax = fig.add_axes([0.15, 0.12, 0.65, 0.03], facecolor=axis_color)
scale_slider = Slider(scale_slider_ax, "Scale", 1.0, 20.0, valinit=scale)

points_slider_ax = fig.add_axes([0.15, 0.07, 0.65, 0.03], facecolor=axis_color)
points_slider = Slider(points_slider_ax, "Points", 3, 10, valinit=points, valfmt="%d")

seeder_slider_ax = fig.add_axes([0.15, 0.02, 0.65, 0.03], facecolor=axis_color)
seeder_slider = Slider(seeder_slider_ax, "Seed", 1, 100, valinit=seeder, valfmt="%d")


def sliders_on_changed(val):
    c = [c0_slider.val, c1_slider.val]
    scale = scale_slider.val
    a = (
        get_random_points(int(seeder_slider.val), n=int(points_slider.val), scale=scale)
        + c
    )
    x, y, _ = get_bezier_curve(a, rad=rad_slider.val, edgy=edgy_slider.val)
    cooordinates = np.array((x, y)).T
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

# -------------------------------------------------------------------- #
save_button_ax = fig.add_axes([0.85, 0.05, 0.1, 0.06])
save_button = Button(save_button_ax, "Save", color="lawngreen", hovercolor="darkgreen")


def save_button_on_clicked(mouse_event):
    # scatter_points.remove()
    ax_bez.axis("off")
    ax_img.axis("off")
    bbox_bez = ax_bez.get_tightbbox(fig.canvas.get_renderer())
    bbox_img = ax_img.get_tightbbox(fig.canvas.get_renderer())
    fig.savefig(
        os.path.join(os.getcwd(), "Bezier"),
        bbox_inches=bbox_bez.transformed(fig.dpi_scale_trans.inverted()),
    )
    fig.savefig(
        os.path.join(os.getcwd(), "Convexhull"),
        bbox_inches=bbox_img.transformed(fig.dpi_scale_trans.inverted()),
    )
    ax_img.axis("on")
    ax_bez.axis("on")


save_button.on_clicked(save_button_on_clicked)
save_button.on_clicked(sliders_on_changed)

# -------------------------------------------------------------------- #
reset_button_ax = fig.add_axes([0.85, 0.12, 0.1, 0.06])
reset_button = Button(
    reset_button_ax, "Reset", color="lawngreen", hovercolor="darkgreen"
)


def reset_button_on_clicked(mouse_event):
    rad_slider.reset()
    edgy_slider.reset()
    c0_slider.reset()
    c1_slider.reset()
    scale_slider.reset()
    points_slider.reset()
    seeder_slider.reset()


reset_button.on_clicked(reset_button_on_clicked)
reset_button.on_clicked(sliders_on_changed)

# -------------------------------------------------------------------- #

background_button_ax = fig.add_axes([0.85, 0.19, 0.1, 0.06])
background_button = Button(
    background_button_ax, "Backgound", color="lawngreen", hovercolor="darkgreen"
)


def background_button_on_clicked(mouse_event):
    global bg_index
    bg_index = bg_index + 1 % (len(BG_LIST))

background_button.on_clicked(background_button_on_clicked)
background_button.on_clicked(sliders_on_changed)

# -------------------------------------------------------------------- #

generate_button_ax = fig.add_axes([0.85, 0.26, 0.1, 0.06])
generate_button = Button(
    generate_button_ax, "Generate", color="lawngreen", hovercolor="darkgreen"
)


def generate_button_on_clicked(mouse_event):
    cluster_image = cluster_makerv2(PATCH_SIZE, png_list, NUM_IMAGES, a_new)
    bg_image = np.array(plt.imread(BG_LIST[bg_index]))
    np.copyto(cluster_image, bg_image, where=cluster_image <= 10)
    ax_img.imshow(cluster_image)


generate_button.on_clicked(sliders_on_changed)
generate_button.on_clicked(generate_button_on_clicked)

# -------------------------------------------------------------------- #
fill_radios_ax = fig.add_axes([0.88, 0.5, 0.1, 0.15], facecolor=axis_color)
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
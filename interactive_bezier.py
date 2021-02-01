from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from bezier import *

bernstein = lambda n, k, t: binom(n, k) * t ** k * (1.0 - t) ** (n - k)
fill_status = False

axis_color = "lightgoldenrodyellow"

fig = plt.figure("Bezier Closed Curve Generator")
fig.suptitle("Interactive Random Bezier Closed Curve Generator", fontsize=12)
ax = fig.add_subplot(111)

fig.subplots_adjust(left=0.3, bottom=0.42)

rad = 0.5
edgy = 0.5

seeder = 1
c = [-1, 1]
scale = 10
points = 4

a = get_random_points(seeder, n=points, scale=scale) + c
x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)


ax.set_xlim([-5, 15])
ax.set_ylim([-5, 15])
ax.set_aspect("equal", "box")
[line] = ax.plot(x, y, linewidth=1, color="k")
scatter_points = ax.scatter(
    a[:, 0],
    a[:, 1],
    color="r",
    marker=".",
    alpha=1,
)

# Sliders
rad_slider_ax = fig.add_axes([0.25, 0.32, 0.65, 0.03], facecolor=axis_color)
rad_slider = Slider(rad_slider_ax, "Radius", 0.0, 1.0, valinit=rad)

edgy_slider_ax = fig.add_axes([0.25, 0.27, 0.65, 0.03], facecolor=axis_color)
edgy_slider = Slider(edgy_slider_ax, "Edginess", 0.0, 5.0, valinit=edgy)

c0_slider_ax = fig.add_axes([0.25, 0.22, 0.65, 0.03], facecolor=axis_color)
c0_slider = Slider(c0_slider_ax, "Move X", -5.0, 5.0, valinit=c[0])

c1_slider_ax = fig.add_axes([0.25, 0.17, 0.65, 0.03], facecolor=axis_color)
c1_slider = Slider(c1_slider_ax, "Move Y", -5.0, 5.0, valinit=c[1])

scale_slider_ax = fig.add_axes([0.25, 0.12, 0.65, 0.03], facecolor=axis_color)
scale_slider = Slider(scale_slider_ax, "Scale", 1.0, 20.0, valinit=scale)

points_slider_ax = fig.add_axes([0.25, 0.07, 0.65, 0.03], facecolor=axis_color)
points_slider = Slider(points_slider_ax, "Points", 1, 20, valinit=points, valfmt="%d")

seeder_slider_ax = fig.add_axes([0.25, 0.02, 0.65, 0.03], facecolor=axis_color)
seeder_slider = Slider(seeder_slider_ax, "Seed", 1, 100, valinit=seeder, valfmt="%d")


def sliders_on_changed(val):
    c = [c0_slider.val, c1_slider.val]
    scale = scale_slider.val
    a = (
        get_random_points(int(seeder_slider.val), n=int(points_slider.val), scale=scale)
        + c
    )
    x, y, _ = get_bezier_curve(a, rad=rad_slider.val, edgy=edgy_slider.val)
    global scatter_points
    if not fill_status:
        ax.clear()
        ax.set_xlim([-5, 15])
        ax.set_ylim([-5, 15])
        ax.set_aspect("equal", "box")
        ax.plot(x, y, linewidth=1, color="k")
    else:
        ax.clear()
        ax.set_xlim([-5, 15])
        ax.set_ylim([-5, 15])
        ax.set_aspect("equal", "box")
        ax.fill(x, y, color="k")
    scatter_points = ax.scatter(
        a[:, 0],
        a[:, 1],
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

save_button_ax = fig.add_axes([0.85, 0.4, 0.1, 0.08])
save_button = Button(save_button_ax, "Save", color="lawngreen", hovercolor="darkgreen")


def save_button_on_clicked(mouse_event):
    scatter_points.remove()
    ax.axis("off")
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    fig.savefig(
        os.path.join(os.getcwd(), "Bezier"),
        bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()),
    )
    ax.axis("on")


save_button.on_clicked(save_button_on_clicked)
save_button.on_clicked(sliders_on_changed)


reset_button_ax = fig.add_axes([0.85, 0.5, 0.1, 0.08])
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

fill_radios_ax = fig.add_axes([0.025, 0.5, 0.15, 0.15], facecolor=axis_color)
fill_radios = RadioButtons(fill_radios_ax, ("No fill", "Fill"), active=0)


def fill_radios_on_clicked(label):
    global fill_status
    if label == "Fill":
        fill_status = True
    else:
        fill_status = False


fill_radios.on_clicked(fill_radios_on_clicked)
fill_radios.on_clicked(sliders_on_changed)
plt.show()
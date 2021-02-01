import numpy as np
import cv2
from tkinter import *
from PIL import Image, ImageTk
from functools import partial
from cluster import *
from convex import *
from convexhull import *

PATH = "../pngs"
PATCH_SIZE = 1000
resize = PATCH_SIZE // 2
no_update = True
png_list = glob.glob(PATH + "/*")


def gen_cluster(num_images=5):
    # grab a reference to the image panels
    global panelA, panelB, panelC

    cluster_image = cluster_maker(PATCH_SIZE, png_list, num_images, no_update)
    convexhull_image, convexhull_image_fill = convex_hull_image(
        cluster_image, thickness=1
    )
    convexhull_cluster_image = convex(image=cluster_image)

    ##conversion bla bla
    cluster_image = cv2.cvtColor(cluster_image, cv2.COLOR_RGBA2BGRA)
    # convexhull_image = cv2.cvtColor(convexhull_image, cv2.COLOR_BGR2RGB)
    convexhull_cluster_image = cv2.cvtColor(convexhull_cluster_image, cv2.COLOR_BGR2RGB)

    # convert the images to PIL format
    cluster_image_show = Image.fromarray(cv2.resize(cluster_image, (resize, resize)))
    convexhull_image_show = Image.fromarray(
        cv2.resize(convexhull_image, (resize, resize))
    )
    convexhull_cluster_image_show = Image.fromarray(
        cv2.resize(convexhull_cluster_image, (resize, resize))
    )

    # ...and then to ImageTk format
    cluster_image_show = ImageTk.PhotoImage(cluster_image_show)
    convexhull_image_show = ImageTk.PhotoImage(convexhull_image_show)
    convexhull_cluster_image_show = ImageTk.PhotoImage(convexhull_cluster_image_show)

    # if the panels are None, initialize them
    if panelA is None or panelB is None or panelC is None:
        # Cluster
        panelA = Label(image=cluster_image_show)
        panelA.image = cluster_image_show
        panelA.pack(side="left", padx=10, pady=10)

        # Convexhull
        panelB = Label(image=convexhull_image_show)
        panelB.image = convexhull_image_show
        panelB.pack(side="right", padx=10, pady=10)

        # Cluster with Convexhull
        panelC = Label(image=convexhull_cluster_image_show)
        panelC.image = convexhull_cluster_image_show
        panelC.pack(side="bottom", padx=10, pady=10)
    # otherwise, update the image panels
    else:
        # update the pannels
        panelA.configure(image=cluster_image_show)
        panelB.configure(image=convexhull_image_show)
        panelC.configure(image=convexhull_cluster_image_show)
        panelA.image = cluster_image_show
        panelB.image = convexhull_image_show
        panelC.image = convexhull_cluster_image_show


# Set up GUI
window = Tk()  # Makes main window
window.wm_title("Cluster Generator")
# window.config(background="#FFFFFF")

panelA = None
panelB = None
panelC = None

# Slider window (slider controls stage position)
# sliderFrame = tk.Frame(window, width=600, height=100)
# sliderFrame.grid(row=600, column=0, padx=10, pady=2)


gen_cluster_num = partial(gen_cluster, num_images=5)
btn = Button(window, text="Generate Cluster", command=gen_cluster_num)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

window.mainloop()  # Starts GUI
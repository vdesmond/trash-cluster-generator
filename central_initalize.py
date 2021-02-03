import numpy as np
import random
import scipy.stats as stats
import glob
from rotateimg import *

PATH = "../pngs"
PATCH_SIZE = 1000
NUM_IMAGES = 5
png_list = glob.glob(PATH + "/*")


def image_placer(img, bg, bezier_coordinates):
    h, w, _ = img.shape
    size = bg.shape[0]

    x_offset = np.random.random()

    y1, y2 = y_offset, y_offset + h
    x1, x2 = x_offset, x_offset + w
    alpha_s = img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        bg[y1:y2, x1:x2, c] = alpha_s * img[:, :, c] + alpha_l * bg[y1:y2, x1:x2, c]
    return (bg, x_offset, y_offset)


def cluster_makerv2(patch_size, png_list, num_images):
    background = np.zeros((patch_size, patch_size, 3), np.uint8)
    result = background.copy()
    imlist = random.sample(png_list, num_images)
    x_offlist, y_offlist = [], []
    for png in imlist:
        img = cv2.imread(png, -1)
        result, x_offset, y_offset = image_placer(img, result)
        x_offlist.append(x_offset)
        y_offlist.append(y_offset)
    return result
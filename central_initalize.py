import numpy as np
import random
import scipy.stats as stats
import glob
from rotateimg import *
import skimage.transform as transform

PATH = "../pngs"
PATCH_SIZE = (720, 1280)
NUM_IMAGES = 1
png_list = glob.glob(PATH + "/*")


def image_placerv2(img, bg, bezier_coordinates, patch_size):
    print(np.count_nonzero(img))
    angle = np.random.randint(-10, 10) * (np.pi / 180.0)  # Convert to radians
    angle = 0
    zoom = np.random.random() * 0.4 + 0.8  # Zoom in range [0.8,1.2)
    zoom = 1

    tform = transform.AffineTransform(
        scale=(zoom, zoom), rotation=angle,  translation=(0, 0))
    img = transform.warp(img, tform.inverse)

    h, w, _ = img.shape
    print(np.count_nonzero(img))
    size = bg.shape[0]
    center_x, center_y = bezier_coordinates[-1][0], bezier_coordinates[-1][1]
    ratio_x, ratio_y = (center_x + 5) / 20, 1 - ((center_y + 5) / 20)
    x_offset = int((np.random.random() * 5) + (ratio_x * patch_size[1]) - (h / 2))
    y_offset = int((np.random.random() * 5) + (ratio_y * patch_size[0]) - (w / 2))
    y1, y2 = y_offset, y_offset + h
    x1, x2 = x_offset, x_offset + w
    alpha_s = img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        bg[y1:y2, x1:x2, c] = alpha_s * img[:, :, c] + alpha_l * bg[y1:y2, x1:x2, c]
    return (bg, x_offset, y_offset)


def cluster_makerv2(patch_size, png_list, num_images, bezier_coordinates):
    background = np.zeros((*patch_size, 3), np.uint8)
    result = background.copy()
    imlist = random.sample(png_list, num_images)
    # x_offlist, y_offlist = [], []
    for png in imlist:
        img = cv2.imread(png, -1)
        result, x_offset, y_offset = image_placerv2(
            img, result, bezier_coordinates, patch_size
        )
        # x_offlist.append(x_offset)
        # y_offlist.append(y_offset)
    return result

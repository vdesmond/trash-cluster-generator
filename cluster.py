import cv2
import numpy as np
import random
import scipy.stats as stats
import glob
from rotateimg import *

PATH = "../pngs"
PATCH_SIZE = 1000
NUM_IMAGES = 5


def get_truncated_normal(mean=500, sd=250, low=0, upp=1000):
    return stats.truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
    ).rvs()


def offset_update(x_offset, y_offset, size, h, w):

    x_offset += random.randint(-size / 10, size / 10)
    y_offset += random.randint(-size / 10, size / 10)
    y1, y2 = y_offset, y_offset + h
    x1, x2 = x_offset, x_offset + w
    x_coords = [x1, x2]
    y_coords = [y1, y2]
    x_constraint = all(0 < e < size for e in x_coords)
    y_constraint = all(0 < e < size for e in y_coords)
    while not x_constraint:
        x_offset += random.randint(-size / 10, size / 10)
        x1, x2 = x_offset, x_offset + w
        x_coords = [x1, x2]
        x_constraint = all(0 < e < size for e in x_coords)
    while not y_constraint:
        y_offset += random.randint(-size / 10, size / 10)
        y1, y2 = y_offset, y_offset + h
        y_coords = [y1, y2]
        y_constraint = all(0 < e < size for e in y_coords)

    # print(x_constraint, y_constraint)
    return x_offset, y_offset, x1, x2, y1, y2


def image_placer(img, bg):

    h, w, _ = img.shape
    size = bg.shape[0]
    # mu, sigma = size / 2, 100
    mean = (size / 2) - 100
    sigma = size / 10

    x_offset = get_truncated_normal(mean, sigma, low=0, upp=size - h)
    x_offset = x_offset.round().astype("int")

    y_offset = get_truncated_normal(mean, sigma, low=0, upp=size - w)
    y_offset = y_offset.round().astype("int")

    y1, y2 = y_offset, y_offset + h
    x1, x2 = x_offset, x_offset + w

    alpha_s = img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        bg[y1:y2, x1:x2, c] = alpha_s * img[:, :, c] + alpha_l * bg[y1:y2, x1:x2, c]
    return (bg, x_offset, y_offset)


def image_saver(result):
    tmp = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(result)
    rgba = [b, g, r, alpha]
    final = cv2.merge(rgba, 4)
    return final


def cluster_maker(patch_size, png_list, num_images, no_update):
    background = np.zeros((patch_size, patch_size, 3), np.uint8)
    result = background.copy()
    imlist = random.sample(png_list, num_images)
    x_offlist, y_offlist = [], []
    for png in imlist:
        img = cv2.imread(png, -1)
        result, x_offset, y_offset = image_placer(img, result)
        x_offlist.append(x_offset)
        y_offlist.append(y_offset)
    final = image_saver(result)

    if no_update:
        return final
    else:
        i = 0
        while i < 5:
            print(i)
            result = background.copy()
            for index, png in enumerate(imlist):
                img = cv2.imread(png, -1)
                result, x_offset, y_offset = update_position(
                    x_offlist[index], y_offlist[index], img, result
                )
            final = image_saver(result)
            cv2.imwrite(f"result{random.randint(1,1000)}.png", final)
            i += 1
        return final


def update_position(x_offset, y_offset, img, bg):

    angle = random.randint(-180, 180)
    rotimg = rotate_image(img, angle)

    h, w, _ = rotimg.shape
    size = bg.shape[0]

    x_offset, y_offset, x1, x2, y1, y2 = offset_update(x_offset, y_offset, size, h, w)

    alpha_s = rotimg[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        bg[y1:y2, x1:x2, c] = alpha_s * rotimg[:, :, c] + alpha_l * bg[y1:y2, x1:x2, c]
    return (bg, x_offset, y_offset)


if __name__ == "__main__":
    no_update = False
    png_list = glob.glob(PATH + "/*")
    cluster_image = cluster_maker(PATCH_SIZE, png_list, NUM_IMAGES)
    cv2.imwrite("result.png", cluster_image)

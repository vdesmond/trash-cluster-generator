#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
from PIL import Image
import numpy as np
import random
import time
import matplotlib.colors
import matplotlib.pyplot as plt
import skimage.transform as transform

foreground_full_list = [
    "/home/desmond/Desktop/taco-dataset/pngs/" + s for s in os.listdir("../pngs")
]

cmp = matplotlib.colors.ListedColormap(
    ["tan", "cyan", "pink", "forestgreen", "blue", "purple", "crimson"]
)


def foregroundAug(foreground):
    # Random rotation, zoom, translation
    angle = np.random.randint(-10, 10) * (np.pi / 180.0)  # Convert to radians
    zoom = np.random.random() * 0.4 + 0.8  # Zoom in range [0.8,1.2)
    t_x = np.random.randint(0, int(foreground.shape[1] / 3))
    t_y = np.random.randint(0, int(foreground.shape[0] / 3))

    tform = transform.AffineTransform(
        scale=(zoom, zoom), rotation=angle, translation=(t_x, t_y)
    )
    foreground = transform.warp(foreground, tform.inverse)
    # Random horizontal flip with 0.5 probability
    if np.random.randint(0, 100) >= 50:
        foreground = foreground[:, ::-1]
    return foreground


def compose(foregrounds, background, init):
    background = Image.fromarray(background)
    bg_w, bg_h = background.size

    # Offset list
    t_x_list = []
    t_y_list = []
    for i in range(len(foregrounds)):
        current_foreground = Image.fromarray((foregrounds[i] * 255).astype(np.uint8))
        img_w, img_h = current_foreground.size

        # Random Offsets
        t_x = np.random.randint(int(-bg_w / 1.5), int(bg_w / 1.5))
        t_y = np.random.randint(int(-bg_h / 8), int(bg_h / 1.5))

        t_x_list.append(t_x)
        t_y_list.append(t_y)

        offset = ((bg_w - img_w + t_x) // 2, (bg_h - img_h + t_y) // 2)
        background.paste(
            current_foreground, offset, current_foreground.convert("RGBA")
        )  # RGBA == RGB alpha channel

    return background, t_x_list, t_y_list


def getForegroundMask(
    foregrounds, background, background_mask, classes_list, t_x_list, t_y_list
):

    background = Image.fromarray(background)
    bg_w, bg_h = background.size

    # 2D mask
    mask_new = background_mask.astype(np.uint8)
    for i in range(len(foregrounds)):
        foregrounds[i] = foregrounds[i] * 255  # Scaling

        # Get current foreground mask
        current_foreground = (
            1 - np.uint8(np.all(foregrounds[i][:, :, :3] == 0, axis=2))
        ) * classes_list[i]

        img_w, img_h = current_foreground.shape
        offset = ((bg_w - img_h + t_x_list[i]) // 2, (bg_h - img_w + t_y_list[i]) // 2)

        roi = np.copy(
            mask_new[offset[1] : offset[1] + img_w, offset[0] : offset[0] + img_h]
        )
        roi_mask = np.logical_and(
            np.logical_or(
                (roi == 0).astype(bool), (roi != classes_list[i]).astype(bool)
            ),
            (current_foreground != 0).astype(bool),
        )
        # Paste current foreground mask over previous mask
        np.copyto(
            mask_new[offset[1] : offset[1] + img_w, offset[0] : offset[0] + img_h],
            current_foreground,
            where=roi_mask,
        )

    return mask_new


def generate_cluster(
    background, background_mask, params, climit, foreground_full_list=foreground_full_list, 
):
    # Cluster limits
    cluster_low_limit,  cluster_high_limit = climit
    foreground_list = random.sample(
        foreground_full_list, random.randint(cluster_low_limit, cluster_high_limit)
    )
    # classes_list = [x.rsplit("/", 2)[-2][-1] for x in foreground_list]
    # classes_list = [int(i) for i in classes_list]

    classes_list = [random.randint(3, 7) for _ in foreground_list]
    init_list = random.sample(params[:-1], len(foreground_list))

    foregrounds = []
    for i in foreground_list:
        foregrounds.append(np.asarray(Image.open(i)))

    for i in range(len(foregrounds)):
        foregrounds[i] = foregroundAug(foregrounds[i])

    try:
        final_background, t_x_list, t_y_list = compose(foregrounds, background, init_list)
        mask_new = getForegroundMask(
            foregrounds,
            background,
            background_mask,
            classes_list,
            t_x_list,
            t_y_list,
        )
        mask_new_pil = Image.fromarray(mask_new)
        return final_background, mask_new, mask_new_pil
    except Exception as e:
        print(e)
        return 0


def save_generate(final_background, mask_new, mask_new_pil):
    savedate = int(time.time() * 10)
    final_background.save(f"./img_{savedate}.jpeg")
    mask_new_pil.save(f"./label_{savedate}.png")
    plt.imsave(
        f"./rgb_label_{savedate}.png",
        np.asarray(mask_new),
        vmin=1,
        vmax=7,
        cmap=cmp,
    )
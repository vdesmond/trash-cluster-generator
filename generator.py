#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
import random
import time
import matplotlib.colors
import matplotlib.pyplot as plt
import skimage.transform as transform
import traceback

from rangemap import translate_offset
from edgecrop import edgecrop

foreground_full_list = [
    "/home/desmond/Desktop/taco-dataset/pngs/" + s for s in os.listdir("../pngs")
]

cmp = matplotlib.colors.ListedColormap(
    ["tan", "cyan", "pink", "forestgreen", "blue", "purple", "crimson"]
)


def foregroundAug(foreground):
    # ! add scale
    # Random rotation, zoom, translation
    angle = np.random.randint(-10, 10) * (np.pi / 180.0)  # Convert to radians
    zoom = np.random.random() * 0.4 + 0.2  # Zoom in range [0.2,0.6)

    t_x, t_y = 0, 0

    tform = transform.AffineTransform(
        scale=(zoom, zoom), rotation=angle, translation=(t_x, t_y)
    )
    foreground = transform.warp(foreground, tform.inverse)
    # Random horizontal flip with 0.5 probability
    if np.random.randint(0, 100) >= 50:
        foreground = foreground[:, ::-1]

    return edgecrop(foreground)


def compose(foregrounds, background, init, center, offset_list):
    background = Image.fromarray(background)
    flipbg = background.transpose(Image.FLIP_TOP_BOTTOM)

    # Offset list
    offset_new_list = []
    for i in range(len(foregrounds)):
        current_foreground = Image.fromarray((foregrounds[i] * 255).astype(np.uint8))
        im_w, im_h = current_foreground.size

        # Quadrant
        current_init = init[i]
        theta = np.arctan2(current_init[1] - center[1], current_init[0] - center[0])
        angle = np.degrees(theta)
        

        # Offset
        offset = offset_list[i]
        print(offset)

        if 0 <= angle <= 90:
            offset_new = offset[0] - im_w, offset[1]
        elif -90 <= angle < 0:
            offset_new = offset[0] - im_w, offset[1] - im_h
        elif -180 <= angle <= 90:
            offset_new = offset[0], offset[1] - ( im_h // 2)
        else:
            offset_new = offset

        flipbg.paste(
            current_foreground, offset, current_foreground.convert("RGBA")
        )  # RGBA == RGB alpha channel
        offset_new_list.append(offset_new)
    background = flipbg.transpose(Image.FLIP_TOP_BOTTOM)

    return background, offset_new_list


def getForegroundMask(
    foregrounds, background, background_mask, classes_list, modified_offs
):

    background = Image.fromarray(background)

    # 2D mask
    mask_new = background_mask.astype(np.uint8)
    for i in range(len(foregrounds)):
        foregrounds[i] = foregrounds[i] * 255  # Scaling

        # Get current foreground mask
        current_foreground = (
            1 - np.uint8(np.all(foregrounds[i][:, :, :3] == 0, axis=2))
        ) * classes_list[i]

        img_w, img_h = current_foreground.shape
        offset = modified_offs[i]

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
    background, background_mask, params, climit, limits, dims, foreground_full_list=foreground_full_list, 
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
    curve_center = params[-1]

    foregrounds = []
    for i in foreground_list:
        foregrounds.append(np.asarray(Image.open(i)))

    for i in range(len(foregrounds)):
        foregrounds[i] = foregroundAug(foregrounds[i])

    offsets = [translate_offset(p, limits, dims) for p in init_list]

    try:
        final_background, modified_offs = compose(foregrounds, background, init_list, curve_center, offsets)
        mask_new = getForegroundMask(
            foregrounds,
            background,
            background_mask,
            classes_list,
            modified_offs
        )
        mask_new_pil = Image.fromarray(mask_new)
        return final_background, mask_new, mask_new_pil
    except Exception as e:
        # print(e)
        traceback.print_exc()
        return 0


def save_generate(final_background, mask_new, mask_new_pil):
    savedate = int(time.time() * 10)
    # bg = final_background.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    final_background.save(f"./img_{savedate}.jpeg")
    mask_new_pil.save(f"./label_{savedate}.png")
    plt.imsave(
        f"./rgb_label_{savedate}.png",
        np.asarray(mask_new),
        vmin=1,
        vmax=7,
        cmap=cmp,
    )
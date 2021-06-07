# -*- coding: utf-8 -*-
import os
import random
import time
import traceback
from collections import deque

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as transform
from PIL import Image

from .gen_utils import edgecrop, init_index_gen, translate_offset

fg_path = os.getcwd() + "/foregrounds"
foreground_full_list = []

for sub_folder in sorted(os.listdir(fg_path)):
    path = os.path.join(fg_path, sub_folder)
    files = os.listdir(path)
    files_path = [os.path.join(path, file) for file in files]
    foreground_full_list.append(files_path)

class_weights = (0.5, 0.3, 0.2)

# ? Beach, Other Background, Glass, Metal, Plastic
cmp = matplotlib.colors.ListedColormap(
    [
        "tan",
        "cyan",
        "pink",
        "forestgreen",
        "blue",
    ]  # ? default colors
)

# cmp = matplotlib.colors.ListedColormap(
#     [
#         "#EBCB8B",
#         "#88C0D0",
#         "#B48EAD",
#         "#A3BE8C",
#         "#BF616A",
#     ]  # ? Banner colors
# )

history = deque()


def foregroundAug(foreground):
    # Random rotation, zoom, translation
    angle = np.random.randint(-10, 10) * (np.pi / 180.0)  # Convert to radians
    zoom = np.random.random() * 0.2 + 0.1  # Zoom in range [0.1,0.3)

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

        if 0 <= angle <= 90:
            offset_new = offset[0] - (im_w // 2), offset[1]
        elif -90 <= angle < 0:
            offset_new = offset[0] - (im_w // 2), offset[1] - (im_h // 2)
        elif -180 <= angle <= 90:
            offset_new = offset[0], offset[1] - (im_h // 2)
        else:
            offset_new = offset

        flipbg.paste(
            current_foreground, offset_new, current_foreground.convert("RGBA")
        )  # RGBA ==> RGB alpha channel

        offset_new_list.append(offset_new)

    background = flipbg.transpose(Image.FLIP_TOP_BOTTOM)

    return background, offset_new_list


def getForegroundMask(
    foregrounds, background, background_mask, classes_list, modified_offs, new_cluster
):

    background = Image.fromarray(background)

    if new_cluster:
        mask_new = np.flipud((background_mask * 255).astype(np.uint8))
    else:
        mask_new = np.flipud(background_mask)

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
                np.logical_or((roi == 1).astype(bool), (roi == 2).astype(bool)),
                (roi != classes_list[i]).astype(bool),
            ),
            (current_foreground != 0).astype(bool),
        )

        # Paste current foreground mask over previous mask
        np.copyto(
            mask_new[offset[1] : offset[1] + img_w, offset[0] : offset[0] + img_h],
            current_foreground,
            where=roi_mask,
        )

    return np.flipud(mask_new)


def generate_cluster(
    background,
    background_mask,
    params,
    climit,
    limits,
    dims,
    foreground_full_list=foreground_full_list,
    new_cluster=True,
):
    if background is None:
        return None, None, None, None

    # ? Cluster limits
    cluster_low_limit, cluster_high_limit = climit

    # ? Get foregrounds
    fg_sampler = np.random.choice(
        len(class_weights),
        random.randint(cluster_low_limit, cluster_high_limit),
        p=class_weights,
    )
    foreground_list = [random.choice(foreground_full_list[c]) for c in fg_sampler]

    classes_list = [x.rsplit("_", 1)[0][-1] for x in foreground_list]
    classes_list = [int(i) for i in classes_list]

    init_indexes = init_index_gen(range(len(params[:-1])), len(foreground_list))
    init_list = np.asarray(params)[init_indexes]
    curve_center = params[-1]

    foreground_images = []
    for i in foreground_list:
        foreground_images.append(np.asarray(Image.open(i)))

    for i in range(len(foreground_images)):
        foreground_images[i] = foregroundAug(foreground_images[i])

    offsets = [translate_offset(p, limits, dims) for p in init_list]

    cache_for_update = (
        background[:],
        background_mask.copy(),
        classes_list,
        foreground_images[:],
        init_indexes,
    )

    try:
        final_background, modified_offs = compose(
            foreground_images, background, init_list, curve_center, offsets
        )

        mask_new = getForegroundMask(
            foreground_images,
            background,
            background_mask,
            classes_list,
            modified_offs,
            new_cluster,
        )
        mask_new_pil = Image.fromarray(mask_new)

        return final_background, mask_new, mask_new_pil, cache_for_update

    except ValueError:
        if new_cluster:
            return None, None, None, None
        return (
                background,
                background_mask,
                Image.fromarray(background_mask),
                cache_for_update,
        )

    except Exception:
        traceback.print_exc()

    finally:
        if cache_for_update[0] is not None:
            if len(history) >= 1 and new_cluster:
                for i in range(len(history)):
                    history.pop()
            history.append(cache_for_update)


def update_cluster(
    background,
    background_mask,
    classes_list,
    foregrounds,
    init_indexes,
    params,
    limits,
    dims,
    new_cluster,
):

    curve_center = params[-1]
    init_list = np.asarray(params)[init_indexes]
    offsets = [translate_offset(p, limits, dims) for p in init_list]

    cache_for_update = (
        background[:],
        background_mask.copy(),
        classes_list,
        foregrounds[:],
        init_indexes,
    )

    try:
        final_background, modified_offs = compose(
            foregrounds, background, init_list, curve_center, offsets
        )
        mask_new = getForegroundMask(
            foregrounds,
            background,
            background_mask,
            classes_list,
            modified_offs,
            new_cluster,
        )
        mask_new_pil = Image.fromarray(mask_new)
        return final_background, mask_new, mask_new_pil, cache_for_update

    except ValueError:
        return (
            background,
            background_mask,
            Image.fromarray(background_mask),
            cache_for_update,
        )

    except Exception:
        traceback.print_exc()

    finally:
        history.pop()
        history.append(cache_for_update)


def undo_func():

    try:
        old_cache = history.pop()
        background, background_mask, *_ = old_cache
        return background, background_mask, Image.fromarray(background_mask), old_cache

    except IndexError:
        return None, None, None, None

    except Exception:
        traceback.print_exc()


def save_generate(final_background, mask_new, mask_new_pil):
    savedate = int(time.time() * 10)
    final_background.save(f"./img_{savedate}.jpeg")
    mask_new_pil.save(f"./label_{savedate}.png")
    plt.imsave(
        f"./rgb_label_{savedate}.png",
        mask_new,
        vmin=1,
        vmax=5,
        cmap=cmp,
    )

# -*- coding: utf-8 -*-
import random

import numpy as np


def edgecrop(img):
    image_data_bw = img.max(axis=2)
    non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
    cropBox = (
        min(non_empty_rows),
        max(non_empty_rows),
        min(non_empty_columns),
        max(non_empty_columns),
    )

    image_data_new = img[cropBox[0] : cropBox[1] + 1, cropBox[2] : cropBox[3] + 1, :]
    return image_data_new


def translate_range(value, fromMin, fromMax, toMin, toMax):
    fromSpan = fromMax - fromMin
    toSpan = toMax - toMin
    valueScaled = float(value - fromMin) / float(fromSpan)
    return toMin + (valueScaled * toSpan)


def translate_offset(offset, limits, dims):
    tr_x = int(translate_range(offset[0], *limits, 0, dims[0]))
    tr_y = int(translate_range(offset[1], *limits, 0, dims[1]))
    return (tr_x, tr_y)


def init_index_gen(fg_list, n_chunks):
    quo, rem = divmod(len(fg_list), n_chunks)
    chunks = (
        fg_list[i * quo + min(i, rem) : (i + 1) * quo + min(i + 1, rem)]
        for i in range(n_chunks)
    )
    return [random.choice(ck) for ck in chunks]

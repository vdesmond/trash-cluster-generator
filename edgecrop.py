#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

# def edgecrop(img):
#     pixels = img.load()
#     xlist = []
#     ylist = []
#     for y in range(0, img.size[1]):
#         for x in range(0, img.size[0]):
#             if pixels[x, y] != (0, 0, 0, 0):
#                 xlist.append(x)
#                 ylist.append(y)
#     left = min(xlist)
#     right = max(xlist)
#     top = min(ylist)
#     bottom = max(ylist)

#     img = img.crop((left-2, top-2, right+2, bottom+2))
#     return img

def edgecrop(img):
    image_data_bw = img.max(axis=2)
    non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

    image_data_new = img[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
    return image_data_new
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from PIL import Image

for file in os.listdir("./bg_labels"):
    im = Image.open(os.path.join(os.getcwd(), "bg_labels", file))
    beach = 0
    sea = 0
    unknown = 0

    for pixel in im.getdata():
        if pixel == 1:
            beach += 1
        elif pixel == 2:
            sea += 1
        else:
            unknown += 1
    print(
        f"Image: {file:15} Beach = {beach}; Other background = {sea}; Unknown ="
        f" {unknown}"
    )

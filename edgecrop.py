#!/usr/bin/env python
# -*- coding: utf-8 -*-

def edgecrop(img):
    pixels = img.load()
    xlist = []
    ylist = []
    for y in range(0, img.size[1]):
        for x in range(0, img.size[0]):
            if pixels[x, y] != (0, 0, 0, 0):
                xlist.append(x)
                ylist.append(y)
    left = min(xlist)
    right = max(xlist)
    top = min(ylist)
    bottom = max(ylist)

    img = img.crop((left-2, top-2, right+2, bottom+2))
    return img
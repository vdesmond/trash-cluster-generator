#!/usr/bin/env python
# -*- coding: utf-8 -*-

def translate_range(value, fromMin, fromMax, toMin, toMax):
    fromSpan = fromMax - fromMin
    toSpan = toMax - toMin
    valueScaled = float(value - fromMin) / float(fromSpan)
    return toMin + (valueScaled * toSpan)

def translate_offset(offset, limits, dims):
    tr_x = int(translate_range(offset[0], *limits, 0, dims[0]))
    tr_y = int(translate_range(offset[1], *limits, 0, dims[1]))
    return (tr_x, tr_y)
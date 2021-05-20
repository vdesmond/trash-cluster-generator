# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom

bernstein = lambda n, k, t: binom(n, k) * t ** k * (1.0 - t) ** (n - k)


def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


class Segment:
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array(
            [self.r * np.cos(self.angle1), self.r * np.sin(self.angle1)]
        )
        self.p[2, :] = self.p2 + np.array(
            [self.r * np.cos(self.angle2 + np.pi), self.r * np.sin(self.angle2 + np.pi)]
        )
        self.curve = bezier(self.p, self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(
            points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw
        )
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def get_bezier_curve(a, rad=0.2, edgy=0):
    """given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy) / np.pi + 0.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x, y = c.T
    return x, y, s


def get_random_points(seeder, n=5, scale=2, mindst=None, rec=0):
    if seeder > 0:
        np.random.seed(seeder)
    else:
        pass
    """create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or 0.7 / n
    a = np.random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1) ** 2)
    if np.all(d >= mindst) or rec >= 200:
        return a * scale
    else:
        return get_random_points(seeder, n=n, scale=scale, mindst=mindst, rec=rec + 1)


def bezier_plot(fill=True, rad=0.5, edgy=0.05):
    c = [0, 1]
    a = get_random_points(seeder=0, n=4, scale=10) + c
    x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
    fig = plt.figure(figsize=(12, 12))
    if fill:
        fig = plt.figure(figsize=(12, 12))
        plt.fill(x, y, "k")
        # plt.plot(x, y, "k")
        plt.axis("off")
        fig.savefig(os.path.join(os.getcwd(), "Bezier-fill"), bbox_inches="tight")
        plt.clf()

    plt.plot(x, y, "k")
    plt.xlim((-5, 15))
    plt.ylim((-2, 18))
    # plt.axis("off")
    fig.savefig(os.path.join(os.getcwd(), f"Bezier"), bbox_inches="tight")


if __name__ == "__main__":
    bezier_plot()

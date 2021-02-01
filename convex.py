import cv2
import numpy as np
import glob
import random


def convex(imgpath=None, image=None):
    if imgpath:
        image = cv2.imread(imgpath)
    else:
        image = image
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    img_raw = img

    contours, hier = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    length = len(contours)
    templist = [contours[i] for i in range(length)]
    cont = np.vstack(templist)
    hull = cv2.convexHull(cont)
    uni_hull = []
    uni_hull.append(hull)
    cv2.drawContours(image, uni_hull, -1, 255, 2)

    cv2.imwrite(f"{random.randint(1,1000)}-cnvx.png", image)
    return image


if __name__ == "__main__":
    for path in glob.glob("*"):
        if path[-3:] == "png":
            convex(imgpath=path)

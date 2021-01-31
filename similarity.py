import cv2
import numpy as np
import similaritymeasures as sms
import matplotlib.pyplot as plt

bezier = cv2.imread("convexhull.png", 0)
bezier = cv2.resize(bezier, (460, 460))

convexhull = cv2.imread("convexhull.png", 0)
convexhull = cv2.resize(convexhull, (460, 460))

bezier_indices = np.where(bezier <= 127)
bezier_coordinates = np.array(list(zip(bezier_indices[0], bezier_indices[1])))

convexhull_indices = np.where(convexhull <= 127)
convexhull_coordinates = np.array(
    list(zip(convexhull_indices[0], convexhull_indices[1]))
)

print(convexhull_coordinates)

metric = sms.pcm(bezier_coordinates, convexhull_coordinates)
print(metric)  # lower value == more similar

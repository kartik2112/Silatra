import cv2
import numpy as np

img = cv2.imread('segmented.jpg')
fast = cv2.FastFeatureDetector_create()
key_points = fast.detect(img, None)
orb = cv2.ORB_create()
key_points, descriptors = orb.compute(img, key_points)
print(descriptors)
img_with_key_points = cv2.drawKeypoints(img, key_points, None, color=(0,255,0), flags=0)
cv2.imshow('Original image, Image with keypoints', np.hstack([img, img_with_key_points]))
cv2.waitKey(15000)
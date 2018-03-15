import numpy as np
import cv2
from matplotlib import pyplot as plt

lower = np.array([0,137,100],np.uint8)
upper = np.array([255,200,150],np.uint8)


# camera = cv2.VideoCapture(0)

# keep looping over the frames in the video
# while True:
    # grab the current frame
    # (grabbed, frame) = camera.read()

    # resize the frame, convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the speicifed upper and lower boundaries

frame = cv2.imread('../../training-images/FaceOcclusion/2.png')
converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
skinMask = cv2.inRange(converted, lower, upper)

# apply a series of erosions and dilations to the mask
# using an elliptical kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
# skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
# skinMask = cv2.erode(skinMask, kernel, iterations = 2)

skinMask = cv2.morphologyEx(skinMask,cv2.MORPH_CLOSE,kernel)
skinMask = cv2.morphologyEx(skinMask,cv2.MORPH_OPEN,kernel)
skinMask = cv2.dilate(skinMask, kernel, iterations = 1)

# blur the mask to help remove noise, then apply the
# mask to the frame
skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
skin = cv2.bitwise_and(frame, frame, mask = skinMask)

cv2.imshow("Img",skin)


img = frame
# img = cv2.imread('coins.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# cv2.imshow('Threshold',thresh)

thresh = skinMask

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

cv2.imshow("Sure FG",sure_fg)
cv2.imshow("Sure BG",sure_bg)
cv2.imshow("Unknown",unknown)


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0


markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

cv2.imshow("Fin",markers)
cv2.imshow("Fin1",img)

cv2.waitKey(0)
# if cv2.waitKey(0) == ord('q'):
#     break
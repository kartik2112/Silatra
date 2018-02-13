import cv2, numpy as np, time, math
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

lower = np.array([0,137,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

DATA_LOC = 'training-images-varun\\Letters\\'

image = cv2.imread(DATA_LOC+'a\\1.png')

HEIGHT, WIDTH, _ = image.shape

blur = cv2.blur(image,(3,3))
ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)

#Create a binary image with where white will be skin colors and rest is black
mask2 = cv2.inRange(ycrcb,lower,upper)
ret,thresh = cv2.threshold(mask2,127,255,0)

_,contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
length = len(contours)
maxArea = -1
if length > 0:
    for i in range(length):  # find the biggest contour (according to area)
        temp = contours[i]
        area = cv2.contourArea(temp)
        if area > maxArea:
            maxArea = area
            ci = i

res = contours[ci]
hull = cv2.convexHull(res)
final_image = np.zeros(image.shape, np.uint8)
cv2.drawContours(final_image, [res], 0, (0, 255, 0), 2)
cv2.drawContours(final_image, [hull], 0, (0, 0, 255), 3)

# Create a ROI
x1,y1,w1,h1 = cv2.boundingRect(hull)
cv2.rectangle(final_image, (x1,y1), (x1+w1,y1+h1), (255,0,0), thickness=2)

(x,y),(major_axis,minor_axis),angle = cv2.fitEllipse(hull)
cv2.ellipse(final_image,(int(x),int(y)),(int(major_axis/2),int(minor_axis/2)),angle,0.0,360.0,(0,255,0),1)

cv2.imshow('res',final_image)
cv2.waitKey(100000)
cv2.destroyAllWindows()

center = (x-x1,y-y1)
eccentricity = (1 - (major_axis/minor_axis) ** 2 ) ** 0.5
hull_area = cv2.contourArea(hull)
scale = hull_area/(WIDTH*HEIGHT)
act_r = ((center[0])**2+(center[1])**2)**0.5
angle = 180-angle if angle > 90 else angle

print('%2d ,%3.3f, %1.3f, %1.3f, %4.3f' % (angle/180, act_r, scale, eccentricity, hull_area))
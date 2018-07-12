import cv2, numpy as np, time, math
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

lower = np.array([0,135,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

DATA_LOC = 'training-images-kartik\\Digits\\0\\551.png'

image = cv2.imread(DATA_LOC)

HEIGHT, WIDTH, _ = image.shape

blur = cv2.blur(image,(3,3))
ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)

#Create a binary image with where white will be skin colors and rest is black
mask2 = cv2.inRange(ycrcb,lower,upper)
ret,thresh = cv2.threshold(mask2,127,255,0)

cv2.imshow('threshed',thresh)
cv2.waitKey(1000000)

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

x1,y1,w1,h1 = cv2.boundingRect(res)
#cv2.rectangle(final_image, (x1,y1), (x1+w1,y1+h1), (255,0,0), thickness=2)

cv2.drawContours(final_image, contours, ci, (255,255,255), cv2.FILLED)
hand = final_image[y1:y1+h1, x1:x1+w1]
cv2.imshow('Hand',hand)
cv2.waitKey(100000)
cv2.destroyAllWindows()
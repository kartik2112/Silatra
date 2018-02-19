# =(b2-min(b:b))/(max(b:b)-min(b:b))
import cv2, numpy as np, time, math
from math import ceil
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

lower = np.array([0,145,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

start = time.time()
grid = (20,20)   #(rows,columns)

data_loc_changed = False
#for label in [chr(ord('a')+i) for i in range(26)]:
try:
    image = cv2.imread('../Gesture recognition\\training-images-tejas\\ThumbsUp\\199.png')

    blur = cv2.blur(image,(3,3))
    ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)

    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(ycrcb,lower,upper)
    _,thresh = cv2.threshold(mask2,127,255,0)

    cv2.imshow('threshold',thresh)
    cv2.waitKey(100000)

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

    x,y,w,h = cv2.boundingRect(contours[ci])
    hand = np.zeros((image.shape[1], image.shape[0], 1), np.uint8)
    cv2.drawContours(hand, contours, ci, 255, cv2.FILLED)
    _,hand = cv2.threshold(hand[y:y+h,x:x+w], 127,255,0)

    cv2.imshow('hand',hand)
    cv2.waitKey(100000)
    cv2.destroyAllWindows()
    HEIGHT, WIDTH = hand.shape
    
    data = [ [0 for haha in range(grid[0])] for hah in range(grid[1]) ]
    h, w = float(HEIGHT/grid[1]), float(WIDTH/grid[0])
    
    for column in range(1,grid[1]+1):
        for row in range(1,grid[0]+1):
            fragment = hand[ceil((column-1)*h):min(ceil(column*h), HEIGHT),ceil((row-1)*w):min(ceil(row*w),WIDTH)]
            _,contour,_ = cv2.findContours(fragment,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            try: area = cv2.contourArea(contour[0])
            except: area=0
            area = area/(h*w)
            data[column-1][row-1] = area
    
    to_write_data = ''
    for column in range(grid[1]):
        for row in range(grid[0]):
            to_write_data += str(data[column][row]) + ','
    print(to_write_data[:-1])
except Exception as e:
    print(e)
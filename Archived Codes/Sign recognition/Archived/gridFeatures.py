import numpy as np
import cv2
import time
from math import ceil
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle

# classifier = pickle.load(open('./Models/KNN_Grid_ModelDump.sav','rb'))
classifier = pickle.load(open('./Models/digits_and_letters_model_new.sav','rb'))
print("Loaded KNN Model")

grid = (20,20)

def extractFeatures(frame):
    _,contours,_ = cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    try:
        length = len(contours)
        print(length)
        maxArea = -1
        ci = -1
        
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                area = cv2.contourArea(contours[i])
                if area > maxArea:
                    maxArea = area
                    ci = i
        if ci == -1:
            return -1
        # res = contours[ci]
        # hull = cv2.convexHull(res)
        # final_image = np.zeros(frame.shape, np.uint8)
        # cv2.drawContours(final_image, [res], 0, (0, 255, 0), 2)
        # cv2.drawContours(final_image, [hull], 0, (0, 0, 255), 3)
        x,y,w,h = cv2.boundingRect(contours[ci])
        
        # hand = np.zeros((frame.shape[1], frame.shape[0], 1), np.uint8)
        # cv2.drawContours(hand, contours, ci, 255, cv2.FILLED)
        
        hand = frame[y:y+h,x:x+w]
        print(hand.shape)
        print(x,y,w,h)
        # ret123,hand = cv2.threshold(hand1,127,255,cv2.THRESH_BINARY)
        # print(ret123)

        (HEIGHT,WIDTH) = hand.shape
        print(HEIGHT,WIDTH)
        
        data = [ [0 for haha in range(grid[0])] for hah in range(grid[1]) ]
        h, w = float(HEIGHT/grid[1]), float(WIDTH/grid[0])
        
        for column in range(1,grid[1]+1):
            for row in range(1,grid[0]+1):
                fragment = hand[ceil((column-1)*h):min(ceil(column*h), HEIGHT),ceil((row-1)*w):min(ceil(row*w),WIDTH)]
                _,contour,_ = cv2.findContours(fragment,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                try: area = cv2.contourArea(contour[0])
                except: area=0.0
                area = float(area/(h*w))
                data[column-1][row-1] = area
                
        features = []
        for column in range(grid[1]):
            for row in range(grid[0]):
                features.append(data[column][row])
        cv2.imshow('Your hand',hand)

        return features

        
    except Exception as e:
        print("DetectSign Error:",e)
        # final_image = np.zeros(frame.shape, np.uint8)
        # cv2.putText(final_image, 'Cannot find hand', (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), thickness=2)
        # cv2.imshow('Original', final_image)
        return []
        #continue


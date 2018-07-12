'''
* ImgReceiver.py is the main function that will setup the socket and after the connection is established (in case of TCP)
* and the socket starts receiving the frames, it will invoke the required modules for processing.
'''


# Reference: https://stackoverflow.com/a/23312964/5370202

import socket
import struct
import atexit
import timeit
import sys
import tkinter
import netifaces as ni

import numpy as np
import cv2
import imutils
import dlib
from imutils import face_utils

# from scipy.fftpack import fft, ifft
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pickle

import silatra  #This module is built using SilatraPythonModuleBuilder
import silatra_utils
sys.path.insert(0, "Modules")
import TimingMod as tm
import PersonStabilizer

# Following modules are used specifically for Gesture recognition
sys.path.insert(0, "Gesture_Modules")
import filter_time_series
import gesture_classify
import directionTracker




mode = "TCP"  # TCP | UDP   # This is the type of socket that this server must create for listening
port = 9001                 # This is the port no to which the server socket is attached


recognitionMode = "SIGN"  # SIGN | GESTURE    # This is the mode of recognition. 
                            # Currently, we have designed the recognition in 2 different modes


noOfFramesCollected = 0     # This is used to keep track of the number of frames received and processed by the server socket


'''
* These variables are used to keep track of times needed by each individual component
'''
start_time, start_time_interFrame = 0, 0



total_captured=601  # This is used as an initial count of frames captured for capturing new frames

minNoOfFramesBeforeGestureRecogStart = 70


# def processImage():


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")



### ------------------- GESTURE handling present here -------------------------------------------------------------
if recognitionMode == "GESTURE":
    classifier = pickle.load(open('./Models/sign_classifier_knn.sav','rb'))
    print("Loaded Gesture Recognition KNN Model")
    observations = []
elif recognitionMode == "SIGN":
    classifier = pickle.load(open('./Models/digits_and_letters_model_new.sav','rb'))
    print("Loaded Sign Recognition KNN Model")



### ---------------------------------Timing here--------------------------------------------------------------------
start_time1 = start_time_interFrame = tm.recordTimings(start_time_interFrame,"INTERFRAME",noOfFramesCollected)
### ---------------------------------Timing here--------------------------------------------------------------------

noOfFramesCollected += 1
silatra_utils.displayTextOnWindow("Frame No",str(noOfFramesCollected))


img_np = cv2.imread("../training-images/samplePic.png")

# img_np = imutils.rotate_bound(img_np,90)
img_np = cv2.resize(img_np,(0,0), fx=0.7, fy=0.7)

# if total_captured >= 50:
#     cv2.imwrite('../training-images/kartik/SampleImages/%d.png'%(total_captured),img_np)
#     total_captured += 1

### ---------------------------------Timing here--------------------------------------------------------------------
start_time = tm.recordTimings(start_time,"IMG_CONVERSION",noOfFramesCollected)
### ---------------------------------Timing here--------------------------------------------------------------------

# cv2.resize(img_np,)

# pred = silatra.findMeTheSign(img_np)
cv2.imwrite('../training-images/kartik/SampleImages/%d_OG.png'%(total_captured),img_np)

gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

maxArea1 = 0
faceRect = -1
foundFace = False

for (i, rect) in enumerate(rects):
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 0, 0), -1)
    if w*h > maxArea1:
        maxArea1 = w*h
        faceRect = (x,y,w,h)
        foundFace = True


        



# mask1, foundFace, faceRect = silatra.segment(img_np)
mask1, _, _ = silatra.segment(img_np)

cv2.imwrite('../training-images/kartik/SampleImages/%d_MaskWOFace.png'%(total_captured),mask1)

### ---------------------------------Timing here--------------------------------------------------------------------
start_time = tm.recordTimings(start_time,"SEGMENT",noOfFramesCollected)
### ---------------------------------Timing here--------------------------------------------------------------------

# cv2.imshow("Mask",mask1)
print("Found face at:",foundFace,"as:",faceRect)

# if foundFace:
#     cv2.rectangle(img_np, (int(faceRect[0]),int(faceRect[1])), (int(faceRect[0]+faceRect[2]),int(faceRect[1]+faceRect[3])), (0,0,255), 2)

cv2.imshow("OG Img",img_np)




PersonStabilizer.stabilize(foundFace,noOfFramesCollected,img_np,faceRect,mask1,total_captured)



### ---------------------------------Timing here--------------------------------------------------------------------
start_time = tm.recordTimings(start_time,"STABILIZE",noOfFramesCollected)
### ---------------------------------Timing here--------------------------------------------------------------------

handFound, hand, contours_of_hand = silatra_utils.get_my_hand(mask1)

if recognitionMode == "SIGN":
    if handFound:
        cv2.imshow("Your hand",hand)
        # cv2.imwrite('../training-images/kartik/SampleImages/%d_hand.png'%(total_captured),img_np)
        features = silatra_utils.extract_features(hand, (20,20))
        pred = silatra_utils.predictSign(classifier,features)
    else:
        pred = -1
    silatra_utils.addToQueue(pred)
    pred = silatra_utils.getConsistentSign()

    # pred = -1
    print("Stable Sign:",pred)

    if pred == -1:
        op1  = "--"+"\r\n"
    else:
        if pred == "2":
            pred = "2 / v"
        op1 = pred+"\r\n"

total_captured += 1





### ---------------------------------Timing here--------------------------------------------------------------------
start_time = tm.recordTimings(start_time,"CLASSIFICATION",noOfFramesCollected)
### ---------------------------------Timing here--------------------------------------------------------------------



### ---------------------------------Timing here--------------------------------------------------------------------
tm.recordTimings(start_time1,"OVERALL",noOfFramesCollected)
### ---------------------------------Timing here--------------------------------------------------------------------




cv2.waitKey(0)
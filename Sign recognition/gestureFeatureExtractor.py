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
import os

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
import FaceEliminator

# Following modules are used specifically for Gesture recognition
sys.path.insert(0, "Gesture_Modules")
import filter_time_series
import gesture_classify
import directionTracker



noOfFramesCollected = 0     # This is used to keep track of the number of frames received and processed by the server socket


'''
* These variables are used to keep track of times needed by each individual component
'''
start_time, start_time_interFrame = 0, 0



total_captured=601  # This is used as an initial count of frames captured for capturing new frames

minNoOfFramesBeforeGestureRecogStart = 0

newGestureStarted = False


# def processImage():


detector = dlib.get_frontal_face_detector()


classifier = pickle.load(open('./Models/gesture_model_10_10.knn.sav','rb'))
print("Loaded Gesture Recognition KNN Model")


kfMapper = {'Up':0,'Right':1,'Left':2,'Down':3,'ThumbsUp':4, 'Sun_Up':5, 'Cup_Open':6, 'Cup_Closed':7}


fileW = open('../HMMTrainer/gestures.csv','w')

subdirss = ['GA','GM','GN']
fullForms = {'GA':'Good Afternoon','GM':'Good Morning','GN':'Good Night'}
for subdir in subdirss:
    pathDir = '../training-images/GestureVideos/'+subdir
    vids = os.listdir(pathDir)
    for vid in vids:
        cap = cv2.VideoCapture(pathDir+'/'+vid)
        observationsOP = []
        observations = []

        while True:
            
            ret,img_np = cap.read()
            if not(ret):
                ### ------------------- GESTURE handling present here -----------------------------------------------------
                print("\n\n---------------Recorded observations------------------\n\n",observations)
                print("\n\n---------------Calling middle filtering layer for compression and noise elimination------------------------\n")
                observations = filter_time_series.filterTS(observations)
                gest12 = gesture_classify.recognize(observations)
                print("\n\nVoila! And the gesture contained in the video is",gest12)
                cap.release()
                print(observationsOP)
                fileW.write(fullForms[subdir])
                for obs in observationsOP:
                    fileW.write(','+str(obs))
                fileW.write('\n')
                break
            
            # noOfFramesCollected += 1
            # silatra_utils.displayTextOnWindow("Frame No",str(noOfFramesCollected))
            

            gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale image
            rects = detector(gray, 1)

            maxArea1 = 0
            faceRect = -1
            foundFace = False

            for (i, rect) in enumerate(rects):
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                # cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 0, 0), -1)
                if w*h > maxArea1:
                    maxArea1 = w*h
                    faceRect = (x,y,w,h)
                    foundFace = True
                    



            # mask1, foundFace, faceRect = silatra.segment(img_np)
            mask1, _, _ = silatra.segment(img_np)
            
            mask1 = FaceEliminator.eliminateFace(mask1, foundFace, faceRect)

            # cv2.imshow("Mask",mask1)
            # print("Found face at:",foundFace,"as:",faceRect)
            
            # if foundFace:
            #     cv2.rectangle(img_np, (int(faceRect[0]),int(faceRect[1])), (int(faceRect[0]+faceRect[2]),int(faceRect[1]+faceRect[3])), (0,0,255), 2)

            # cv2.imshow("OG Img",img_np)

            handFound, hand, contours_of_hand = silatra_utils.get_my_hand(mask1)

            if handFound:
                # cv2.imshow("Your hand",hand)
                direction = directionTracker.trackDirection(contours_of_hand)
                print('Frame %3d -> %-11s'%(noOfFramesCollected,direction))
                if direction == "None":                
                    features = silatra_utils.extract_features(hand, (10,10))
                    predicted_sign = silatra_utils.predictSign(classifier,features)
                    # silatra_utils.displayTextOnWindow("Sign",predicted_sign,10,100,1)
                    observations.append((predicted_sign,'None'))
                    observationsOP.append(kfMapper[predicted_sign])
                else:
                    # silatra_utils.displayTextOnWindow("Sign",direction,25,100,1.5)
                    observations.append(('None',direction))
                    observationsOP.append(kfMapper[direction])
            
            
            k = cv2.waitKey(10)
            if k == 'q':
                break
            # elif k=='c':
    
    



cv2.destroyAllWindows()


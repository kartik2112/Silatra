# Reference: https://stackoverflow.com/a/23312964/5370202

# This program reads all the training videos specified using variables subdirss, pathDir
# and finds and saves the hand pose, motion feature set in csv file. This will be used for training of HMM



import sys
import os

import numpy as np
import cv2
import imutils
import dlib
from imutils import face_utils

# from scipy.fftpack import fft, ifft
from sklearn.neighbors import KNeighborsClassifier
import pickle

# import silatra  #This module is built using SilatraPythonModuleBuilder
sys.path.insert(0, "../SiLaTra_Server")
import silatra_utils
sys.path.insert(0, "../SiLaTra_Server/Modules")
import FaceEliminator

# Following modules are used specifically for Gesture recognition
sys.path.insert(0, "../SiLaTra_Server/Gesture_Modules")
import directionTracker


def segment(src_img):
    """
    ### Segment skin areas from hand using a YCrCb mask.

    This function returns a mask with white areas signifying skin and black areas otherwise.

    Returns: mask
    """

    import cv2
    from numpy import array, uint8

    blurred_img = cv2.GaussianBlur(src_img,(5,5),0)
    blurred_img = cv2.medianBlur(blurred_img,5)
    
    blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2YCrCb)

    lower = array([0,137,100], uint8)
    upper = array([255,200,150], uint8)
    mask = cv2.inRange(blurred_img, lower, upper)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


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


# Change the filename to "your" features file
# classifier = pickle.load(open('./SampleKNNModels/silatra_gesture_signs_pregenerated_sample.sav','rb'))
classifier = pickle.load(open('./TempKNNModels/silatra_gesture_signs_pregenerated_sample.sav','rb'))
print("Loaded Gesture Recognition KNN Model")


kfMapper = {'Up':0,'Right':1,'Left':2,'Down':3,'ThumbsUp':4, 'Sun_Up':5, 'Cup_Open':6, 'Cup_Closed':7, 'Apple_Finger':8, 'OpenPalmHori':9, 'Leader_L':10, 'Fist':11, 'That_Is_Good_Circle':12}


# This is an incorrect temporary file so as to prevent ruining the sample file.
# If you understand this and find this, you will be able to change it.
fileW = open('./TempFeatureSetFiles/silatra_gestures_temp.csv','a')

part1 = True
if part1 == True:
    subdirss = ['After','All The Best','Apple','I Am Sorry','Leader','Please Give Me Your Pen','Strike','That is Good','Towards']
else:
    subdirss = ['GN','GM','GA']
fullForms = {'GA':'Good Afternoon','GM':'Good Morning','GN':'Good Night'}
for subdir in subdirss:
    pathDir = '../Dataset/Gesture_Videos_Dataset/'+subdir
    vids = os.listdir(pathDir)
    for vid in vids:
        cap = cv2.VideoCapture(pathDir+'/'+vid)
        observationsOP = []
        observations = []

        while True:
            
            ret,img_np = cap.read()
            if not(ret):
                cap.release()
                print(observations)
                print(observationsOP)
                if part1 == True:
                    fileW.write(subdir)
                else:
                    fileW.write(fullForms[subdir])
                for obs in observationsOP:
                    fileW.write(','+str(obs))
                fileW.write('\n')
                break
            

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
                    
            # mask1, _, _ = silatra.segment(img_np)
            mask1 = segment(img_np)
            
            mask1 = FaceEliminator.eliminateFace(mask1, foundFace, faceRect)

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


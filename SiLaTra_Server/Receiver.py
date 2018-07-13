'''
* Receiver.py is the main function that will setup the socket and after the connection is established (in case of TCP)
* and the socket starts receiving the frames, it will invoke the required modules for processing.
*
* To invoke this file in background, use command:
* (python3 Receiver.py --portNo 49165 --displayWindows False > /dev/null &)
'''


# Reference: https://stackoverflow.com/a/23312964/5370202

import socket
import struct
import atexit
import timeit
import sys
import netifaces as ni
import os
import distutils

import argparse


import numpy as np
import cv2
import imutils
import dlib
from imutils import face_utils

from sklearn.neighbors import KNeighborsClassifier
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))

# import silatra_cpp  #This module is built using SilatraPythonModuleBuilder
import silatra_utils
sys.path.insert(0, dir_path+"/Modules")
import TimingMod as tm
import PersonStabilizer
import FaceEliminator

# Following modules are used specifically for Gesture recognition
sys.path.insert(0, dir_path+"/Gesture_Modules")
import directionTracker
import hmmGestureClassify

currentModuleName = __file__.split(os.path.sep)[-1]

parser = argparse.ArgumentParser(description='Main Entry Point')
parser.add_argument('--portNo', 
                    help='Usage: python3 Receiver.py --portNo 12345')
parser.add_argument('--displayWindows', 
                    help='Usage: python3 Receiver.py --displayWindows True | False')
parser.add_argument('--recognitionMode', 
                    help='Usage: python3 Receiver.py --recognitionMode SIGN | GESTURE')
parser.add_argument('--socketTimeOutEnable', 
                    help='Usage: python3 Receiver.py --socketTimeOutEnable True | False')
parser.add_argument('--stabilize', 
                    help='Usage: python3 Receiver.py --stabilize True | False')
parser.add_argument('--recordVideos', 
                    help='Usage: python3 Receiver.py --recordVideos True | False --subDir GN')
parser.add_argument('--subDir', 
                    help='Usage: python3 Receiver.py --recordVideos True | False --subDir GN')
args = parser.parse_args()



gridSize = (10,10)



port = 49164                 # This is the port no to which the server socket is attached


recognitionMode = "SIGN"  # SIGN | GESTURE    # This is the mode of recognition. 
                            # Currently, we have designed the recognition in 2 different modes

if args.recognitionMode != None and args.recognitionMode in ('SIGN','GESTURE'):
    recognitionMode = args.recognitionMode


if args.stabilize == None:
    stabilizeEnabled = False
else:
    stabilizeEnabled = bool(distutils.util.strtobool(args.stabilize))


if args.socketTimeOutEnable == None:
    socketTimeOutEnable = False
else:
    socketTimeOutEnable = bool(distutils.util.strtobool(args.socketTimeOutEnable))



noOfFramesCollected = 0     # This is used to keep track of the number of frames received and processed by the server socket


'''
* These variables are used to keep track of times needed by each individual component
'''
start_time, start_time_interFrame = 0, 0



total_captured=601  # This is used as an initial count of frames captured for capturing new frames

minNoOfFramesBeforeGestureRecogStart = 70

newGestureStarted = False

lastMsgSentOut = '--\r\n'



detector = dlib.get_frontal_face_detector()

videoCounter = 1
if args.recordVideos == None:
    recordVideos = False
else:
    subdir = args.subDir
    mainDir = '../training-images/GestureVideos/'+subdir
    recordVideos = args.recordVideos
    if not(os.path.isdir(mainDir)):
        os.makedirs(mainDir)

if args.displayWindows == None:
    displayWindows = True
else:
    displayWindows = bool(distutils.util.strtobool(args.displayWindows))


def videoInitializer():
    global videoCounter
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(mainDir+'/Gesture_'+subdir+'_'+'%03d'%(videoCounter)+'.avi',fourcc, 5, (336,448))
    videoCounter += 1
    return out

### ------------------- GESTURE handling present here -------------------------------------------------------------
if recognitionMode == "GESTURE":
    classifier = pickle.load(open(dir_path+'/Models/silatra_gesture_signs.sav','rb'))
    print("Loaded Gesture Recognition KNN Model")
    observations = []
    if recordVideos:
        out = videoInitializer()
    op1 = "Wait..."+"\r\n"
elif recognitionMode == "SIGN":
    classifier = pickle.load(open(dir_path+'/Models/silatra_digits_and_letters_10_10.sav','rb'))
    print("Loaded Sign Recognition KNN Model")



def port_initializer():
    global port
    port = int(port_entry.get())
    opening_window.destroy()


if args.portNo == None:
    port = int(input("Enter port no: "))
else:
    port = int(args.portNo)

# Reference: https://stackoverflow.com/a/24196955/5370202
ni.ifaddresses('wlo1')
ipAddr = ni.ifaddresses('wlo1')[ni.AF_INET][0]['addr']


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
if socketTimeOutEnable:
    s.settimeout(20)
print("TCP Socket successfully created")
s.bind(('', port))
print("TCP Socket binded to %s: %s" %(ipAddr,port))
s.listen(1)
print("Socket is listening")
client, addr = s.accept()     
print('Got TCP connection from', addr)
if socketTimeOutEnable:
    s.settimeout(10)



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



while True:
    
    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time1 = start_time_interFrame = tm.recordTimings(start_time_interFrame,"INTERFRAME",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------

    noOfFramesCollected += 1
    if displayWindows:
        silatra_utils.displayTextOnWindow("Frame No",str(noOfFramesCollected))
    
    
    buf = client.recv(4)

    # print(buf)
    size = struct.unpack('!i', buf)[0]  
    #Reference: https://stackoverflow.com/a/37601966/5370202, https://docs.python.org/3/library/struct.html
    # print(size)
    print("receiving image of size: %s bytes" % size)

    if(size == 0 and recognitionMode == "SIGN"):
        op1 = "QUIT\r\n"
        client.send(op1.encode('ascii'))
        break
    elif(size == 0 and recognitionMode == "GESTURE"):
        ### ------------------- GESTURE handling present here -----------------------------------------------------
        if len(observations) > 0:
            hmmGest12 = hmmGestureClassify.classifyGestureByHMM(observations)
            if displayWindows:
                silatra_utils.displayTextOnWindow("HMMGesture",hmmGest12[0],10,100,1)
            
            print("\n\nVoila! And the gesture recognized by HMM is",hmmGest12)
            op1 = hmmGest12[0] + "\r\n"
            client.send(op1.encode('ascii'))
        op1 = "QUIT\r\n"
        client.send(op1.encode('ascii'))
        break

    data = client.recv(size,socket.MSG_WAITALL)  #Reference: https://www.binarytides.com/receive-full-data-with-the-recv-socket-function-in-python/

    

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time1,"DATA_TRANSFER",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------


    # Instead of storing the image as mentioned in the 1st reference: https://stackoverflow.com/a/23312964/5370202
    # we can directly convert it to Opencv Mat format
    # Reference: https://stackoverflow.com/a/17170855/5370202
    nparr = np.fromstring(data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_np = imutils.rotate_bound(img_np,90)
    img_np = cv2.resize(img_np,(0,0), fx=0.7, fy=0.7)
    


    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time,"IMG_CONVERSION",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------



    # mask1, _, _ = silatra_cpp.segment(img_np)
    mask1 = segment(img_np)

    
        
    
    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time,"SEGMENT",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------
    

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    maxArea1 = 0
    faceRect = -1
    foundFace = False

    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        if w*h > maxArea1:
            maxArea1 = w*h
            faceRect = (x,y,w,h)
            foundFace = True

            
    mask1 = FaceEliminator.eliminateFace(mask1, foundFace, faceRect)
    if displayWindows:
        cv2.imshow("Mask12",mask1)

    

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time,"FACEHIDING",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------
    

    if displayWindows:
        cv2.imshow("OG Img",img_np)

    if stabilizeEnabled:
        PersonStabilizer.stabilize(foundFace,noOfFramesCollected,img_np,faceRect,mask1)

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time,"STABILIZE",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------

    handFound, hand, contours_of_hand = silatra_utils.get_my_hand(mask1)

    if recognitionMode == "SIGN":
        if handFound:
            if displayWindows:
                cv2.imshow("Your hand",hand)
                
            features = silatra_utils.extract_features(hand, gridSize)
            pred = silatra_utils.predictSign(classifier,features)
        else:
            pred = -1
        silatra_utils.addToQueue(pred)
        pred = silatra_utils.getConsistentSign(displayWindows)

        # pred = -1
        print("Stable Sign:",pred)

        if pred == -1:
            op1  = "--"+"\r\n"
        else:
            if pred == "2":
                pred = "2 / v"
            op1 = pred+"\r\n"


    elif recognitionMode == "GESTURE":
        if handFound:
            if displayWindows:
                cv2.imshow("Your hand",hand)
            direction = directionTracker.trackDirection(contours_of_hand)
            print('Frame %3d -> %-11s'%(noOfFramesCollected,direction))
            if direction == "None":                
                features = silatra_utils.extract_features(hand, gridSize)
                predicted_sign = silatra_utils.predictSign(classifier,features)
                if displayWindows:
                    silatra_utils.displayTextOnWindow("Sign",predicted_sign,10,100,1)
                if noOfFramesCollected > minNoOfFramesBeforeGestureRecogStart:
                    if newGestureStarted == False:
                        newGestureStarted = True
                    observations.append((predicted_sign,'None'))
                    if recordVideos:
                        out.write(img_np)
            else:
                if displayWindows:
                    silatra_utils.displayTextOnWindow("Sign",direction,25,100,1.5)
                if noOfFramesCollected > minNoOfFramesBeforeGestureRecogStart and newGestureStarted == True:
                    observations.append(('None',direction))
                    if recordVideos:
                        out.write(img_np)
        else:
            if len(observations)>5:
                ### ------------------- GESTURE handling present here -----------------------------------------------------
                hmmGest12 = hmmGestureClassify.classifyGestureByHMM(observations)
                if displayWindows:
                    silatra_utils.displayTextOnWindow("HMMGesture",hmmGest12[0],10,100,1)
                
                print("\n\nVoila! And the gesture recognized by HMM is",hmmGest12)
                op1 = hmmGest12[0] + "\r\n"
                client.send(op1.encode('ascii'))
                observations = []
                newGestureStarted = False
                if recordVideos:
                    out.release()
                    out = videoInitializer()
            elif len(observations)>0:
                print("Observation sequence too small")
                pass
            else:
                print("New gesture not yet started")
                

        if noOfFramesCollected == minNoOfFramesBeforeGestureRecogStart - 10:
            op1 = "Model ready to recognize\r\n"
        elif noOfFramesCollected == minNoOfFramesBeforeGestureRecogStart:
            op1 = "Start gesture\r\n"
        elif len(observations) == 0:
            pass
        # elif observations[-1][0] == "None":
        #     op1 = observations[-1][1]+"\r\n"
        # else:
        #     op1 = observations[-1][0]+"\r\n"

    else:
        break
    
    

    

    if recognitionMode =="SIGN":
        client.send(op1.encode('ascii'))
        lastMsgSentOut = op1
    elif recognitionMode == "GESTURE" or lastMsgSentOut != op1:
        client.send(op1.encode('ascii'))
        lastMsgSentOut = op1
    

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time,"CLASSIFICATION",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------



    ### ---------------------------------Timing here--------------------------------------------------------------------
    tm.recordTimings(start_time1,"OVERALL",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------



    
    k = cv2.waitKey(10)
    if k == 'q':
        break
    
    




print('Stopped TCP server of port: '+str(port))
print(recognitionMode+" recognition stopped")
tm.displayAllTimings(noOfFramesCollected)




s.close()
cv2.destroyAllWindows()






def cleaners():
    s.close()
    cv2.destroyAllWindows()

atexit.register(cleaners)
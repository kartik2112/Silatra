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

import argparse

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
import hmmGestureClassify

parser = argparse.ArgumentParser(description='Main Entry Point')
parser.add_argument('--recordVideos', 
                    help='Usage: python3 ImgReceiver.py --recordVideos True --subDir GN')
parser.add_argument('--subDir', 
                    help='Usage: python3 ImgReceiver.py --recordVideos True --subDir GN')
args = parser.parse_args()




mode = "TCP"  # TCP | UDP   # This is the type of socket that this server must create for listening
port = 9001                 # This is the port no to which the server socket is attached


recognitionMode = "GESTURE"  # SIGN | GESTURE    # This is the mode of recognition. 
                            # Currently, we have designed the recognition in 2 different modes


noOfFramesCollected = 0     # This is used to keep track of the number of frames received and processed by the server socket


'''
* These variables are used to keep track of times needed by each individual component
'''
start_time, start_time_interFrame = 0, 0



total_captured=601  # This is used as an initial count of frames captured for capturing new frames

minNoOfFramesBeforeGestureRecogStart = 70

newGestureStarted = False


# def processImage():


detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")

videoCounter = 5
subdir = args.subDir
mainDir = '../training-images/GestureVideos/GN1'
if args.recordVideos == None:
    recordVideos = False
else:
    recordVideos = args.recordVideos
    if not(os.path.isdir(mainDir)):
        os.makedirs(mainDir)

def videoInitializer():
    global videoCounter
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(mainDir+'/Gesture_'+subdir+'_'+'%03d'%(videoCounter)+'.avi',fourcc, 5, (336,448))
    videoCounter += 1
    return out

### ------------------- GESTURE handling present here -------------------------------------------------------------
if recognitionMode == "GESTURE":
    classifier = pickle.load(open('./Models/gesture_model_10_10.knn.sav','rb'))
    print("Loaded Gesture Recognition KNN Model")
    observations = []
    if recordVideos:
        out = videoInitializer()
    op1 = "--"+"\r\n"
elif recognitionMode == "SIGN":
    classifier = pickle.load(open('./Models/digits_and_letters_10_10.sav','rb'))
    print("Loaded Sign Recognition KNN Model")



def port_initializer():
    global port
    port = int(port_entry.get())
    opening_window.destroy()



opening_window = tkinter.Tk()
port_label = tkinter.Label(opening_window, text = "Port to be reserved:")
port_label.pack(side = tkinter.LEFT)
port_entry = tkinter.Entry(opening_window, bd=3)
port_entry.pack(side = tkinter.RIGHT)
save_button = tkinter.Button(opening_window, command = port_initializer)
save_button.pack()
opening_window.mainloop()


# Reference: https://stackoverflow.com/a/24196955/5370202
ni.ifaddresses('wlo1')
ipAddr = ni.ifaddresses('wlo1')[ni.AF_INET][0]['addr']

if mode == "TCP":
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         
    print("TCP Socket successfully created")
    s.bind(('', port))        
    print("TCP Socket binded to %s: %s" %(ipAddr,port))
    s.listen(1)     
    print("Socket is listening")
    client, addr = s.accept()     
    print('Got TCP connection from', addr)
else:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)         
    print("UDP Socket successfully created")
    s.bind(('',port))        
    print("UDP Socket binded to %s: %s" %(ipAddr,port))
    UDP_IP_ADDRESS2 = ""
    UDP_SEND_PORT_NO = 0

while True:
    
    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time1 = start_time_interFrame = tm.recordTimings(start_time_interFrame,"INTERFRAME",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------

    noOfFramesCollected += 1
    silatra_utils.displayTextOnWindow("Frame No",str(noOfFramesCollected))
    
    if mode == "TCP":
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
            print("\n\n---------------Recorded observations------------------\n\n",observations)
            print("\n\n---------------Calling middle filtering layer for compression and noise elimination------------------------\n")
            observations = filter_time_series.filterTS(observations)
            gest12 = gesture_classify.recognize(observations)
            print("\n\nVoila! And the gesture contained in the video is",gest12)
            # op1 = "GESTURE:"+gest12 + "\r\n"
            op1 = gest12 + "\r\n"
            client.send(op1.encode('ascii'))
            op1 = "QUIT\r\n"
            client.send(op1.encode('ascii'))
            break

        data = client.recv(size,socket.MSG_WAITALL)  #Reference: https://www.binarytides.com/receive-full-data-with-the-recv-socket-function-in-python/

    else:
        data, addr = s.recvfrom(65507)
        print("Received %d bytes image (UDP Packet) from"%len(data), addr)
        UDP_IP_ADDRESS2,UDP_SEND_PORT_NO = addr[0],addr[1]

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time1,"DATA_TRANSFER",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------

    # if ctr123 % 5 != 0:
    #     continue


    # with open('tst.jpeg', 'wb') as img:
    #         img.write(data)


    # Instead of storing this image as mentioned in the 1st reference: https://stackoverflow.com/a/23312964/5370202
    # we can directly convert it to Opencv Mat format
    #Reference: https://stackoverflow.com/a/17170855/5370202
    nparr = np.fromstring(data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_np = imutils.rotate_bound(img_np,90)
    img_np = cv2.resize(img_np,(0,0), fx=0.7, fy=0.7)
    

    # if total_captured >= 50:
    #     cv2.imwrite('../training-images/kartik/SampleImages/%d.png'%(total_captured),img_np)
    #     total_captured += 1

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time,"IMG_CONVERSION",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------

    # cv2.resize(img_np,)
    
    # pred = silatra.findMeTheSign(img_np)

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
    
    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time,"SEGMENT",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------
    
    mask1 = FaceEliminator.eliminateFace(mask1, foundFace, faceRect)
    cv2.imshow("Mask12",mask1)

    # cv2.imshow("Mask",mask1)
    print("Found face at:",foundFace,"as:",faceRect)
    
    # if foundFace:
    #     cv2.rectangle(img_np, (int(faceRect[0]),int(faceRect[1])), (int(faceRect[0]+faceRect[2]),int(faceRect[1]+faceRect[3])), (0,0,255), 2)

    cv2.imshow("OG Img",img_np)

    # PersonStabilizer.stabilize(foundFace,noOfFramesCollected,img_np,faceRect,mask1)

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time,"STABILIZE",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------

    handFound, hand, contours_of_hand = silatra_utils.get_my_hand(mask1)

    if recognitionMode == "SIGN":
        if handFound:
            cv2.imshow("Your hand",hand)
            features = silatra_utils.extract_features(hand, (10,10))
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


    elif recognitionMode == "GESTURE":
        if handFound:
            cv2.imshow("Your hand",hand)
            direction = directionTracker.trackDirection(contours_of_hand)
            print('Frame %3d -> %-11s'%(noOfFramesCollected,direction))
            if direction == "None":                
                features = silatra_utils.extract_features(hand, (10,10))
                predicted_sign = silatra_utils.predictSign(classifier,features)
                silatra_utils.displayTextOnWindow("Sign",predicted_sign,10,100,1)
                if noOfFramesCollected > minNoOfFramesBeforeGestureRecogStart:
                    if newGestureStarted == False:
                        newGestureStarted = True
                    observations.append((predicted_sign,'None'))
                    if recordVideos:
                        out.write(img_np)
            else:
                silatra_utils.displayTextOnWindow("Sign",direction,25,100,1.5)
                if noOfFramesCollected > minNoOfFramesBeforeGestureRecogStart and newGestureStarted == True:
                    observations.append(('None',direction))
                    if recordVideos:
                        out.write(img_np)
        else:
            if len(observations)>0:
                ### ------------------- GESTURE handling present here -----------------------------------------------------
                print("\n\n---------------Recorded observations------------------\n\n",observations)
                print("\n\n---------------Calling middle filtering layer for compression and noise elimination------------------------\n")
                hmmGest12 = hmmGestureClassify.classifyGestureByHMM(observations)
                observations = filter_time_series.filterTS(observations)
                gest12 = gesture_classify.recognize(observations)
                silatra_utils.displayTextOnWindow("HMMGesture",hmmGest12[0],10,100,1)
                # silatra_utils.displayTextOnWindow("Gesture",gest12,10,100,1)
                print("\n\nVoila! And the gesture contained in the video is",gest12)
                print("\n\nVoila! And the gesture recognized by HMM is",hmmGest12)
                # op1 = "GESTURE:"+gest12 + "\r\n"
                op1 = gest12 + "\r\n"
                client.send(op1.encode('ascii'))
                observations = []
                newGestureStarted = False
                if recordVideos:
                    out.release()
                    out = videoInitializer()
            else:
                print("New gesture not yet started")
                

        if len(observations) == 0:
            pass
        elif observations[-1][0] == "None":
            op1 = observations[-1][1]+"\r\n"
        else:
            op1 = observations[-1][0]+"\r\n"

    else:
        break
    
    

    
    if mode == "TCP":
        client.send(op1.encode('ascii'))
    else:
        Message = str.encode("Hello")
        clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        clientSock.sendto(Message, (UDP_IP_ADDRESS2, UDP_SEND_PORT_NO))
        print("Sending data")

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time,"CLASSIFICATION",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------



    ### ---------------------------------Timing here--------------------------------------------------------------------
    tm.recordTimings(start_time1,"OVERALL",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------



    
    k = cv2.waitKey(10)
    if k == 'q':
        break
    # elif k=='c':
    
    




print('Stopped '+mode+' server of port: '+str(port))
print(recognitionMode+" recognition stopped")
tm.displayAllTimings(noOfFramesCollected)




# client.close()
s.close()
cv2.destroyAllWindows()






def cleaners():
    s.close()
    cv2.destroyAllWindows()

atexit.register(cleaners)
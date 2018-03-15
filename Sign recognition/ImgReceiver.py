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

import numpy as np
import cv2
import imutils

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
port = 49164                # This is the port no to which the server socket is attached


recognitionMode = "SIGN"  # SIGN | GESTURE    # This is the mode of recognition. 
                            # Currently, we have designed the recognition in 2 different modes


noOfFramesCollected = 0     # This is used to keep track of the number of frames received and processed by the server socket


'''
* These variables are used to keep track of times needed by each individual component
'''
start_time, start_time_interFrame = 0, 0



total_captured=601  # This is used as an initial count of frames captured for capturing new frames

minNoOfFramesBeforeGestureRecogStart = 140


# def processImage():






### ------------------- GESTURE handling present here -------------------------------------------------------------
if recognitionMode == "GESTURE":
    classifier = pickle.load(open('./Models/sign_classifier_knn.sav','rb'))
    print("Loaded Gesture Recognition KNN Model")
    observations = []
elif recognitionMode == "SIGN":
    classifier = pickle.load(open('./Models/digits_and_letters_model_new.sav','rb'))
    print("Loaded Sign Recognition KNN Model")




if mode == "TCP":
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         
    print("TCP Socket successfully created")
    s.bind(('', port))        
    print("TCP Socket binded to %s" %(port))
    s.listen(1)     
    print("Socket is listening")
    client, addr = s.accept()     
    print('Got TCP connection from', addr)
else:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)         
    print("UDP Socket successfully created")
    s.bind(('',port))        
    print("UDP Socket binded to %s" %(port))


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

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time,"IMG_CONVERSION",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------

    # cv2.resize(img_np,)
    
    # pred = silatra.findMeTheSign(img_np)
    mask1, foundFace, faceRect = silatra.segment(img_np)
    
    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time,"SEGMENT",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------

    # cv2.imshow("Mask",mask1)
    print("Found face at:",foundFace,"as:",faceRect)
    
    # if foundFace:
    #     cv2.rectangle(img_np, (int(faceRect[0]),int(faceRect[1])), (int(faceRect[0]+faceRect[2]),int(faceRect[1]+faceRect[3])), (0,0,255), 2)

    cv2.imshow("OG Img",img_np)

    PersonStabilizer.stabilize(foundFace,noOfFramesCollected,img_np,faceRect,mask1)

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time,"STABILIZE",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------

    handFound, hand, contours_of_hand = silatra_utils.get_my_hand(mask1)

    if recognitionMode == "SIGN":
        if handFound:
            cv2.imshow("Your hand",hand)
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
            op1 = chr(pred)+"\r\n"


    elif recognitionMode == "GESTURE":
        if handFound:
            cv2.imshow("Your hand",hand)
            direction = directionTracker.trackDirection(contours_of_hand)
            print('Frame %3d -> %-11s'%(noOfFramesCollected,direction))
            if direction == "None":
                features = silatra_utils.extract_features(hand, (20,20))
                predicted_sign = silatra_utils.predictSign(classifier,features)
                silatra_utils.displayTextOnWindow("Sign",predicted_sign,10,100)
                if noOfFramesCollected > minNoOfFramesBeforeGestureRecogStart:
                    observations.append((predicted_sign,'None'))
            else:
                silatra_utils.displayTextOnWindow("Sign",direction,25,100)
                if noOfFramesCollected > minNoOfFramesBeforeGestureRecogStart:
                    observations.append(('None',direction))

        if len(observations) == 0:
            op1 = "--"+"\r\n"
        elif observations[-1][0] == "None":
            op1 = observations[-1][1]+"\r\n"
        else:
            op1 = observations[-1][0]+"\r\n"

    else:
        break
    
    

    
    if mode == "TCP":
        client.send(op1.encode('ascii'))
    else:
        Message = bytearray([1,2,3,4,5])
        clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        clientSock.sendto(Message, (str(addr), port))

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = tm.recordTimings(start_time,"CLASSIFICATION",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------



    ### ---------------------------------Timing here--------------------------------------------------------------------
    tm.recordTimings(start_time1,"OVERALL",noOfFramesCollected)
    ### ---------------------------------Timing here--------------------------------------------------------------------



    
    k = cv2.waitKey(10)
    if k == 'q':
        break
    elif k=='c':
        if total_captured >= 300: break
        cv2.imwrite('../training-images/tejas/ThumbsUp/%d.png'%(total_captured))
        total_captured += 1
    




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
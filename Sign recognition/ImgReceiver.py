'''
* ImgReceiver.py is the main function that will setup the socket and after the connection is established (in case of TCP)
* and the socket starts receiving the frames, it will invoke the required modules for processing.
'''


# Reference: https://stackoverflow.com/a/23312964/5370202

import socket
import struct
import atexit
import timeit

import numpy as np
import cv2
import imutils

import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pickle

import silatra  #This module is built using SilatraPythonModuleBuilder
import gridFeatures as gridF
import silatra_utils

# Following modules are used specifically for Gesture recognition
import Gesture_Modules/filter_time_series
import Gesture_Modules/gesture_classify


'''
* For stabilization of person, KCF Tracker is used. Now for this face detected using Haar is used as ROI.
* Sometimes the KCF Tracker fails to locate the ROI. For this purpose, even after waiting for `maxNoOfFramesNotTracked` frames,
*   if the tracker fails to locate the ROI, the tracker is reinitialized.
'''
faceStabilizerMode = "OFF"  # This is used to enable/disable the stabilizer using KCF Tracker
trackingStarted = False     # This is used to indicate whether tracking has started or not
noOfFramesNotTracked = 0    # This indicates the no of frames that has not been tracked
maxNoOfFramesNotTracked = 15 # This is the max no of frames that if not tracked, will restart the tracker algo



mode = "TCP"  # TCP | UDP   # This is the type of socket that this server must create for listening
port = 49164                # This is the port no to which the server socket is attached


recognitionMode = "SIGN"  # SIGN | GESTURE    # This is the mode of recognition. 
                            # Currently, we have designed the recognition in 2 different modes


'''
* These variables form part of the logic that is used for stabilizing the stream of signs
* From a stream of most recent `maxQueueSize` signs, the sign that has occured most frequently 
*   with frequency > `minModality` is considered as the consistent sign
'''
preds = []          # This is used as queue for keeping track of last `maxQueueSize` signs for finding out the consistent sign
maxQueueSize = 15   # This is the max size of queue `preds`
noOfSigns = 128     # This is the domain of the values present in the queue `preds`
minModality = int(maxQueueSize/2)   # This is the minimum number of times a sign must be present in `preds` to be declared as consistent
noOfFramesCollected = 0     # This is used to keep track of the number of frames received and processed by the server socket


'''
* These variables are used to keep track of times needed by each individual component
'''
start_time, start_time_interFrame = 0, 0
minTimes,maxTimes,avgTimes = {}, {}, {}
timeKeys = ["OVERALL","DATA_TRANSFER","IMG_CONVERSION","SEGMENT","STABILIZE","CLASSIFICATION","INTERFRAME"]
timeStrings = {
    "OVERALL": "Overall:",
    "DATA_TRANSFER": "   Waiting + Data Transfer:",
    "IMG_CONVERSION": "   Image Conversion:",
    "SEGMENT": "   Segmentation:",
    "STABILIZE": "   Stabilizer",
    "CLASSIFICATION": "   Classification:",
    "INTERFRAME": "   Inter-frame difference"
}
for key12 in timeStrings.keys():
    minTimes[key12] = 100
    avgTimes[key12] = 0.0
    maxTimes[key12] = 0


total_captured=601  # This is used as an initial count of frames captured for capturing new frames




# def processImage():


tracker = cv2.TrackerKCF_create()


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
    start_time_interFrame = silatra_utils.recordTimings(start_time_interFrame,"INTERFRAME")
    start_time1 = timeit.default_timer()
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

        if(size == 0):
            op1 = "QUIT\r\n"
            client.send(op1.encode('ascii'))
            break

        data = client.recv(size,socket.MSG_WAITALL)  #Reference: https://www.binarytides.com/receive-full-data-with-the-recv-socket-function-in-python/

    else:
        data, addr = s.recvfrom(65507)
        print("Received %d bytes image (UDP Packet) from"%len(data), addr)

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = silatra_utils.recordTimings(start_time1,"DATA_TRANSFER")
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
    start_time = silatra_utils.recordTimings(start_time,"IMG_CONVERSION")
    ### ---------------------------------Timing here--------------------------------------------------------------------

    # cv2.resize(img_np,)
    
    # pred = silatra.findMeTheSign(img_np)
    mask1, foundFace, faceRect = silatra.segment(img_np)
    
    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = silatra_utils.recordTimings(start_time,"SEGMENT")
    ### ---------------------------------Timing here--------------------------------------------------------------------

    # cv2.imshow("Mask",mask1)
    print("Found face at:",foundFace,"as:",faceRect)
    
    # if foundFace:
    #     cv2.rectangle(img_np, (int(faceRect[0]),int(faceRect[1])), (int(faceRect[0]+faceRect[2]),int(faceRect[1]+faceRect[3])), (0,0,255), 2)

    cv2.imshow("OG Img",img_np)

    if not(trackingStarted) and foundFace and noOfFramesCollected >= 100:
        trackingStarted = True
        ok = tracker.init(img_np, faceRect)
        trackerInitFace = faceRect
    elif trackingStarted:
        ok, bbox = tracker.update(img_np)
        if ok:
            cv2.rectangle(img_np, (int(bbox[0]),int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (255,0,0), 2)
            
            cv2.imshow("OG Img",img_np)
            rows,cols,_ = img_np.shape
            tx = int(trackerInitFace[0] - bbox[0])
            ty = int(trackerInitFace[1] - bbox[1])
            shiftMatrix = np.float32([[1,0,tx],[0,1,ty]])
            
            # Reference: https://www.docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
            img_np = cv2.warpAffine(img_np,shiftMatrix,(cols,rows))
            mask1 = cv2.warpAffine(mask1,shiftMatrix,(cols,rows))

            cv2.imshow("Stabilized Image",img_np)
            noOfFramesNotTracked = 0
            # cv2.imshow("Stabilized Mask",mask1)
        else:
            noOfFramesNotTracked += 1
            if noOfFramesNotTracked > maxNoOfFramesNotTracked:
                trackingStarted = False
                noOfFramesNotTracked = 0

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = silatra_utils.recordTimings(start_time,"STABILIZE")
    ### ---------------------------------Timing here--------------------------------------------------------------------


    features = gridF.extractFeatures(mask1)

    pred = gridF.predictSign(features)
    
    silatra_utils.addToQueue(pred)

    pred = silatra_utils.getConsistentSign()

    # pred = -1
    print("Stable Sign:",pred)

    if pred == -1:
        op1  = "--"+"\r\n"
    else:
        op1 = chr(pred)+"\r\n"
    

    
    if mode == "TCP":
        client.send(op1.encode('ascii'))
    else:
        Message = bytearray([1,2,3,4,5])
        clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        clientSock.sendto(Message, (str(addr), port))

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = silatra_utils.recordTimings(start_time,"CLASSIFICATION")
    ### ---------------------------------Timing here--------------------------------------------------------------------



    ### ---------------------------------Timing here--------------------------------------------------------------------
    silatra_utils.recordTimings(start_time1,"OVERALL")
    ### ---------------------------------Timing here--------------------------------------------------------------------



    
    k = cv2.waitKey(10)
    if k == 'q':
        break
    elif k=='c':
        if total_captured >= 300: break
        cv2.imwrite('../training-images/tejas/ThumbsUp/%d.png'%(total_captured))
        total_captured += 1
    




print('Stopped server')
print('\n\nTimings for %d frames'%noOfFramesCollected)
for key12 in timeKeys: 
    print(timeStrings[key12])
    print('          Min Time taken:',"%.4fs"%minTimes[key12])
    print('          Avg Time taken:',"%.4fs"%avgTimes[key12])
    print('          Max Time taken:',"%.4fs"%maxTimes[key12])




# client.close()
s.close()
cv2.destroyAllWindows()






def cleaners():
    s.close()
    cv2.destroyAllWindows()

atexit.register(cleaners)
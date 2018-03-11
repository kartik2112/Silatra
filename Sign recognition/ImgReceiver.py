# Reference: https://stackoverflow.com/a/23312964/5370202

import socket
import struct

import numpy as np
import cv2
import imutils

# import test

import atexit
import timeit


import silatra

import detectSign_SK as dsk

import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pickle

faceStabilizerMode = "OFF"
tracker = cv2.TrackerKCF_create()


mode = "TCP"  # TCP | UDP
port = 49164

preds = []
maxQueueSize = 15
noOfSigns = 128
minModality = int(maxQueueSize/2)

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

noOfFramesCollected = 0



def addToQueue(pred):
    global preds, maxQueueSize, minModality, noOfSigns
    if len(preds) == maxQueueSize:
        preds = preds[1:]
    preds += [pred]
    

def predictSign():
    global preds, maxQueueSize, minModality, noOfSigns
    modePrediction = -1
    countModality = minModality

    if len(preds) == maxQueueSize:
        countPredictions = [0]*noOfSigns

        for pred in preds:
            if pred != -1:
                countPredictions[pred]+=1
        
        for i in range(noOfSigns):
            if countPredictions[i]>countModality:
                modePrediction = i
                countModality = countPredictions[i]

        displaySignOnImage(modePrediction)
    
    return modePrediction

def displayTextOnWindow(windowName,textToDisplay):
    signImage = np.zeros((200,200,1),np.uint8)

    cv2.putText(signImage,textToDisplay,(75,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,8);

    cv2.imshow(windowName,signImage);

def displaySignOnImage(predictSign):
    dispSign = "--"
    if predictSign != -1:
        dispSign = chr(predictSign)+"";

    displayTextOnWindow("Prediction",dispSign)
    
def recordTimings(start_time,time_key):
    global minTimes,maxTimes,avgTimes,noOfFramesCollected
    if noOfFramesCollected != 0: 
        elapsed = timeit.default_timer() - start_time
        avgTimes[time_key] = avgTimes[time_key] * ((noOfFramesCollected-1)/noOfFramesCollected) + elapsed/noOfFramesCollected
        minTimes[time_key] = elapsed if elapsed < minTimes[time_key] else minTimes[time_key]
        maxTimes[time_key] = elapsed if elapsed > maxTimes[time_key] else maxTimes[time_key]
    return timeit.default_timer()



# def processImage():



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

total_captured=601
trackingStarted = False
noOfFramesNotTracked = 0
while True:
    
    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time_interFrame = recordTimings(start_time_interFrame,"INTERFRAME")
    start_time1 = timeit.default_timer()
    ### ---------------------------------Timing here--------------------------------------------------------------------

    noOfFramesCollected += 1
    displayTextOnWindow("Frame No",str(noOfFramesCollected))
    
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
    start_time = recordTimings(start_time1,"DATA_TRANSFER")
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
    start_time = recordTimings(start_time,"IMG_CONVERSION")
    ### ---------------------------------Timing here--------------------------------------------------------------------

    # cv2.resize(img_np,)
    
    # pred = silatra.findMeTheSign(img_np)
    mask1, foundFace, faceRect = silatra.segment(img_np)
    
    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = recordTimings(start_time,"SEGMENT")
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
            if noOfFramesNotTracked > 15:
                trackingStarted = False
                noOfFramesNotTracked = 0

    ### ---------------------------------Timing here--------------------------------------------------------------------
    start_time = recordTimings(start_time,"STABILIZE")
    ### ---------------------------------Timing here--------------------------------------------------------------------


    pred = dsk.findSign(mask1)

    print("Received Sign:",pred)
    addToQueue(pred)

    pred = predictSign()
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
    start_time = recordTimings(start_time,"CLASSIFICATION")
    ### ---------------------------------Timing here--------------------------------------------------------------------



    ### ---------------------------------Timing here--------------------------------------------------------------------
    recordTimings(start_time1,"OVERALL")
    ### ---------------------------------Timing here--------------------------------------------------------------------



    # test.testMe(img_np)
    k = cv2.waitKey(10)
    if k == 'q':
        break
    elif k=='c':
        if total_captured is 300: break
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
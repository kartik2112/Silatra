# Reference: https://stackoverflow.com/a/23312964/5370202

import socket
import struct

import numpy as np
import cv2

# import test

import atexit


import silatra

import detectSign_SK as dsk

import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pickle


preds = []
maxQueueSize = 15
noOfSigns = 128
minModality = int(maxQueueSize/2)


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

def displaySignOnImage(predictSign):
    dispSign = "--"
    if predictSign != -1:
        dispSign = chr(predictSign)+"";

    signImage = np.zeros((200,200,1),np.uint8)

    cv2.putText(signImage,dispSign,(75,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,8);

    cv2.imshow("Prediction",signImage);
    




# next create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)         
print("Socket successfully created")
 
# reserve a port on your computer in our
# case it is 12345 but it can be anything
port = 49164
 
# Next bind to the port
# we have not typed any ip in the ip field
# instead we have inputted an empty string
# this makes the server listen to requests 
# coming from other computers on the network
s.bind(('', port))        
print("socket binded to %s" %(port))
 
# put the socket into listening mode
s.listen(1)     
print("socket is listening")
 
# a forever loop until we interrupt it or 
# an error occurs
client, addr = s.accept()     
print('Got connection from', addr)

# address = ("10.0.0.12", 5000)
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind(address)
# s.listen(1000)


# client, addr = s.accept()
# print 'got connected from', addr
# buf = ''
# ctr = 4
# while ctr>0:
#     buf += str(client.recv(1))
#     ctr-=1

# ctr123 = 0
while True:
    # ctr123 += 1
    buf = client.recv(4)
    # print(buf)
    size = struct.unpack('!i', buf)[0]  
    #Reference: https://stackoverflow.com/a/37601966/5370202, https://docs.python.org/3/library/struct.html
    # print(size)
    print("receiving image of size: %s bytes" % size)

    if(size == 0):
        break

    data = client.recv(size,socket.MSG_WAITALL)  #Reference: https://www.binarytides.com/receive-full-data-with-the-recv-socket-function-in-python/

    # if ctr123 % 5 != 0:
    #     continue


    # with open('tst.jpeg', 'wb') as img:
    #         img.write(data)


    # Instead of storing this image as mentioned in the 1st reference: https://stackoverflow.com/a/23312964/5370202
    # we can directly convert it to Opencv Mat format
    #Reference: https://stackoverflow.com/a/17170855/5370202
    nparr = np.fromstring(data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cv2.imshow("Img",img_np)

    # cv2.resize(img_np,)
    
    # pred = silatra.findMeTheSign(img_np)
    img1 = silatra.segment(img_np)
    cv2.imshow("asdf",img1)

    pred = dsk.findSign(img1)

    print("Received Sign:",pred)
    addToQueue(pred)

    pred = predictSign()
    # pred = -1
    print("Stable Sign:",pred)
    if pred == -1:
        op1  = "--"+"\r\n"
    else:
        op1 = chr(pred)+"\r\n"
    client.send(op1.encode('ascii'))


    # test.testMe(img_np)

    if cv2.waitKey(10) == 'q':
        break
    


print('received, yay!')

client.close()
cv2.closeAllWindows()


def cleaners():
    client.close()
    cv2.closeAllWindows()

atexit.register(cleaners)
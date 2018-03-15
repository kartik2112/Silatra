# Reference: https://stackoverflow.com/a/23312964/5370202

import socket
import struct

import numpy as np
import cv2
import imutils

# import test

import atexit


import silatra
from utils import segment, get_my_hand, extract_features
import filter_time_series
import gesture_classify

# import detectSign_SK as dsk

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
    


contour_start=False
f = open('bounds.txt')
param = int(f.read().strip())
f.close()

lower = np.array([0,param,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

classifier = pickle.load(open('sign_classifier_knn.sav','rb'))
''' data = pd.read_csv('gesture_data.csv')
X = data[['f'+str(i) for i in range(400)]].values
Y = data['label'].values
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, Y)
pickle.dump(classifier, open('sign_classifier_knn.sav','wb')) '''

prev_x, prev_y = 0, 0
THRESHOLD, frame_n = 20, 1

observations = []






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
        print(observations)

        print("Calling middle filtering layer for compression and noise elimination:")

        observations = filter_time_series.filterTS(observations)
        gest12 = gesture_classify.recognize(observations)
        print("Voila! And the gesture contained in the video is",gest12)
        # op1 = "GESTURE:"+gest12 + "\r\n"
        op1 = gest12 + "\r\n"
        client.send(op1.encode('ascii'))
        op1 = "QUIT\r\n"
        client.send(op1.encode('ascii'))
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

    img_np = imutils.rotate_bound(img_np,90)

    cv2.imshow("Img",img_np)

    # cv2.resize(img_np,)
    
    # pred = silatra.findMeTheSign(img_np)
    # img1 = silatra.segment(img_np)
    # cv2.imshow("asdf",img1)

    # pred = dsk.findSign(img1)

    # print("Received Sign:",pred)
    # addToQueue(pred)

    # pred = predictSign()
    # # pred = -1
    # print("Stable Sign:",pred)


    try:
        mask = silatra.segment(img_np)
        _,thresh = cv2.threshold(mask,127,255,0)

        hand_contour = get_my_hand(thresh, return_only_contour=True)
        # hull = cv2.convexHull(hand_contour)
        # final_image = np.zeros(img_np.shape, np.uint8)
        # cv2.drawContours(final_image, [hand_contour], 0, (0, 255, 0), 2)
        
        M = cv2.moments(hand_contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if prev_x is 0 and prev_y is 0: prev_x, prev_y = 0, 0
        
        delta_x, delta_y, slope, direction = prev_x-cx, prev_y-cy, 0, 'None'

        if delta_x**2+delta_y**2 > THRESHOLD**2:
            if delta_x is 0 and delta_y > 0: slope = 999 # inf
            elif delta_x is 0 and delta_y < 0: slope = -999 # -inf
            else: slope = float(delta_y/delta_x)

            if slope < 1.0 and slope >= -1.0 and delta_x > 0: direction = 'Right'
            elif slope < 1.0 and slope >= -1.0: direction = 'Left'
            elif (slope >= 1.0 or slope <=-1.0) and delta_y > 0.0: direction = 'Up'
            elif slope >= 1.0 or slope <=-1.0: direction = 'Down'

            THRESHOLD = 7
            prev_x, prev_y = cx, cy
            observations.append(('None',direction))
        else:
            # Classification here
            hand = get_my_hand(mask)
            features = extract_features(hand, (20,20))
            predicted_sign = classifier.predict([features])[0]
            observations.append((predicted_sign,'None'))
            direction = 'No movement'
            THRESHOLD = 20
        
        print('Frame %3d -> %-11s'%(frame_n,direction))

        #cv2.imshow('Tracking hands', final_image)
        frame_n += 1
        
        #print('Time per frame: '+str((time.time()-start_time)*1000)+'ms\r',end='')


        if observations[-1][0] == "None":
            op1 = observations[-1][1]+"\r\n"
        else:
            op1 = observations[-1][0]+"\r\n"

        # if pred == -1:
        #     op1  = "--"+"\r\n"
        # else:
        #     op1 = chr(pred)+"\r\n"
        client.send(op1.encode('ascii'))


        # test.testMe(img_np)
    except Exception as e:
        print(e)

    if cv2.waitKey(10) == 'q':
        break
    


print('received, yay!')

# client.close()
s.close()
cv2.destroyAllWindows()


def cleaners():
    s.close()
    cv2.destroyAllWindows()

atexit.register(cleaners)
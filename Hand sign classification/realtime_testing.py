import cv2, numpy as np, time
from math import ceil
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('new_data.csv')

X = data[['f'+str(i) for i in range(400)]].values
Y = data['label'].values

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, Y)

#Open Camera object
cap = cv2.VideoCapture(0)
cap.set(3,640); cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS, 20)

f = open('bounds.txt')
param = int(f.read().strip())
f.close()

contour_start=False
lower = np.array([0,param,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

grid = (20,20)

for c in [str(i) for i in range(10)]+[chr(ord('a')+i) for i in range(26) if i not in [7,9,21]]: print('%-4s'%(c),end=' ')
print('\n')
while(1):
    start_time = time.time()
    ret, frame = cap.read()
    HEIGHT, WIDTH, _ = frame.shape

    x,y,w,h = 100,100,300,300
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    roi = frame[y:y+h,x:x+w]

    blur = cv2.blur(roi,(3,3))
    ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)

    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(ycrcb,lower,upper)
    ret,thresh = cv2.threshold(mask2,127,255,0)

    #Find contours of the filtered frame
    _,contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if not contour_start:
        cv2.imshow('Skin segmentation using YCrCb mask',mask2)
        cv2.imshow('Original',frame)
    else:
        try:
            length = len(contours)
            maxArea = -1
            if length > 0:
                for i in range(length):  # find the biggest contour (according to area)
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i

                res = contours[ci]
                hull = cv2.convexHull(res)
                final_image = np.zeros(frame.shape, np.uint8)
                cv2.drawContours(final_image, [res], 0, (0, 255, 0), 2)
                cv2.drawContours(final_image, [hull], 0, (0, 0, 255), 3)
                
            x,y,w,h = cv2.boundingRect(contours[ci])
            hand = np.zeros((frame.shape[1], frame.shape[0], 1), np.uint8)
            cv2.drawContours(hand, contours, ci, 255, cv2.FILLED)
            _,hand = cv2.threshold(hand[y:y+h,x:x+w], 127,255,0)

            HEIGHT, WIDTH = hand.shape
            
            data = [ [0 for haha in range(grid[0])] for hah in range(grid[1]) ]
            h, w = float(HEIGHT/grid[1]), float(WIDTH/grid[0])
            
            for column in range(1,grid[1]+1):
                for row in range(1,grid[0]+1):
                    fragment = hand[ceil((column-1)*h):min(ceil(column*h), HEIGHT),ceil((row-1)*w):min(ceil(row*w),WIDTH)]
                    _,contour,_ = cv2.findContours(fragment,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    try: area = cv2.contourArea(contour[0])
                    except: area=0.0
                    area = float(area/(h*w))
                    data[column-1][row-1] = area
            
            features = []
            for column in range(grid[1]):
                for row in range(grid[0]):
                    features.append(data[column][row])
            cv2.imshow('Your hand',hand)
            cv2.imshow('Original',frame)

            predictions = classifier.predict_proba([features]).tolist()[0]
            for prob in predictions: print('%.2f'%(prob),end=' ')
            print('',end='\r')
        except Exception as e:
            print(e,end='\r')
            final_image = np.zeros(frame.shape, np.uint8)
            cv2.putText(final_image, 'Cannot find hand', (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), thickness=2)
            cv2.imshow('Original', final_image)
            #continue

    k = cv2.waitKey(50) & 0xFF
    if k == ord('q'):
        break
    elif k==ord('s'):
        contour_start=not contour_start
        cv2.destroyAllWindows()
        
cap.release()
cv2.destroyAllWindows()
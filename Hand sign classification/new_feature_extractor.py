import cv2, numpy as np, time, math, winsound
from math import ceil
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

start = time.time()
dump_file = open('digits_and_letters.csv','w')
grid = (20,20)   #(rows,columns)
for i in range(grid[0]*grid[1]): dump_file.write('f'+str(i)+',')
dump_file.write('label\n')

total_images_parsed = 0
DATA_LOCS = ['training-images-tejas\\Digits\\', 'training-images-kartik\\Letters\\', 'training-images-tejas\\Backup\\Digits\\', 'training-images-kartik\\Digits\\', 'training-images-varun\\Digits\\']
params = [145,145,145,135,137]
for loc in range(len(DATA_LOCS)):
    DATA_LOC = DATA_LOCS[loc]
    lower = np.array([0,params[loc],60],np.uint8)
    upper = np.array([255,180,127],np.uint8)
    
    for label in [str(i) for i in range(10)]+[chr(ord('a')+i) for i in range(26)]:
        for i in range(1,700):
            try:
                print(' '*160+'\rProcessing image: %3d, Label = %c, From Location: %s' % (i,label,DATA_LOC),end='\r')
                #image = cv2.imread(DATA_LOC+label+'\\'+str(i)+'.png')
                image = cv2.imread(DATA_LOC+str(label)+"\\"+str(i)+'.png')

                if image.shape[0] == 0: continue
                blur = cv2.blur(image,(3,3))
                ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)

                #Create a binary image with where white will be skin colors and rest is black
                mask2 = cv2.inRange(ycrcb,lower,upper)
                _,thresh = cv2.threshold(mask2,127,255,0)

                _,contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                length = len(contours)
                maxArea = -1
                if length > 0:
                    for i in range(length):  # find the biggest contour (according to area)
                        temp = contours[i]
                        area = cv2.contourArea(temp)
                        if area > maxArea:
                            maxArea = area
                            ci = i

                x,y,w,h = cv2.boundingRect(contours[ci])
                # hand = np.zeros((image.shape[1], image.shape[0], 1), np.uint8)
                # cv2.drawContours(hand, contours, ci, 255, cv2.FILLED)
                # _,hand = cv2.threshold(hand[y:y+h,x:x+w], 127,255,0)
                hand = thresh[y:y+h,x:x+w]

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
                
                to_write_data = ''
                for column in range(grid[1]):
                    for row in range(grid[0]):
                        to_write_data += str(data[column][row]) + ','
                if label in [chr(ord('a')+i) for i in range(26)]: to_write_data += str(ord(label)-ord('a')+10) + '\n'
                else: to_write_data += str(label) + '\n'
                dump_file.write(to_write_data)
                total_images_parsed += 1
            except Exception as e:
                #print(e)
                continue
total = (time.time() - start)
print(' '*160+'\rTotal time required = %3.3fs' % (total))
print('Total images parsed: %d'%(total_images_parsed))
winsound.Beep(1000, 100)
winsound.Beep(1200, 100)
winsound.Beep(1500, 100)
winsound.Beep(1700, 100)
# This program reads all the training images specified using variables DATA_LOCS, TRAINING_LABELS
# and finds and saves the grid feature set in csv file. This will be used for training of KNN

import cv2, numpy as np, time, math
import silatra
from math import ceil
import os.path


start = time.time()


# Set the mode to a (append) or w (write) depending on the need


# This is an incorrect temporary file so as to prevent ruining the sample file.
# If you understand this and find this, you will be able to change it.
dump_file = open('./TempFeatureSetFiles/silatra_gesture_signs.csv','w')
# dump_file = open('./TempFeatureSetFiles/silatra_digits_letters.csv','a')




grid = (10,10)   #(rows,columns)




print('Labels: ',end='\r')
print([str(i) for i in range(10)]+[chr(ord('a')+i) for i in range(26)]+['Cup_Closed','Cup_Open','Sun_Up','ThumbsUp'])
total_images_parsed = 0



# Gesture Signs data should be stored in 1 csv file. Example: gestures_pregenerated_sample.csv

DATA_LOCS = ['../Dataset/Hand_Poses_Dataset/Gesture_Signs/']
TRAINING_LABELS = ['Leader_L','Apple_Finger','Cup_Closed','Cup_Open','ThumbsUp','Sun_Up','Fist','OpenPalmHori','That_Is_Good_Circle','That_Is_Good_Point']




# Digits and letters data should be stored together in 1 csv file. Example: silatra_signs_pregenerated_sample

# DATA_LOCS = ['../Dataset/Hand_Poses_Dataset/Digits/']
# TRAINING_LABELS = ['0','1','2','3','4','5','6','7','8','9']

# DATA_LOCS = ['../Dataset/Hand_Poses_Dataset/Letters/']
# TRAINING_LABELS = [chr(ord('a')+i) for i in range(26)]


for loc in range(len(DATA_LOCS)):
    DATA_LOC = DATA_LOCS[loc]
    # lower = np.array([0,params[loc],60],np.uint8)
    # upper = np.array([255,180,127],np.uint8)
    
    for label in TRAINING_LABELS:
        for i in range(1,1200+1):
            try:
                if not(os.path.isfile(DATA_LOC+str(label)+"/"+str(i)+'.png')):
                    # skip if image not present. This is possible if some images are manually deleted if they look like outlier data
                    continue
                print(' '*120+'\rProcessing image: %3d, Label = %s, From Location: %s' % (i,label,DATA_LOC+str(label)+"\\"+str(i)+'.png'),end='\r')
                
                image = cv2.imread(DATA_LOC+str(label)+"/"+str(i)+'.png')

                if image.shape[0] == 0: continue



                # blur = cv2.blur(image,(3,3))
                # ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)
                # #Create a binary image with where white will be skin colors and rest is black
                # mask = cv2.inRange(ycrcb,lower,upper)
                # open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                # close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
                # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
                # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)


                # Replacement for above block. If this silatra module is not installed using 
                #   python3 setup.py install 
                # from inside of SilatraPythonModuleBuilder, then uncomment above block of code
                mask,_,_ = silatra.segment(image)

                _,thresh = cv2.threshold(mask,127,255,0)

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
                
                to_write_data += str(label) + '\n'
                # cv2.imshow('Hand',hand)
                dump_file.write(to_write_data)
                total_images_parsed += 1
                # cv2.waitKey(30)
            except Exception as e:
                print(e)
                continue

dump_file.close()
total = (time.time() - start)
print(' '*160+'\rTotal time required = %3.3fs' % (total))
print('Total images parsed: %d'%(total_images_parsed))
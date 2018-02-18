# =(b2-min(b:b))/(max(b:b)-min(b:b))
import cv2, numpy as np, time, math,winsound
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

lower = np.array([0,137,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

start = time.time()
dump_file = open('data.csv','w')
dump_file.write('angle_90,r,area_ellipse,area_contour,scale,eccentricity,norm_r,norm_area_ellipse,norm_area_contour,label\n')

print([str(i) for i in range(10)]+[chr(ord('a')+i) for i in range(26)])

total_images_parsed = 0
DATA_LOCS = ['training-images-tejas\\Digits\\', 'training-images-kartik\\Letters\\', 'training-images-tejas\\Backup\\Digits\\', 'training-images-kartik\\Digits\\', 'training-images-varun\\Digits\\']
params = [145,145,145,135,137]
for loc in range(len(DATA_LOCS)):
    lower = np.array([0,params[loc],60],np.uint8)
    upper = np.array([255,180,127],np.uint8)
    for label in [str(i) for i in range(10)]+[chr(ord('a')+i) for i in range(26)]:
        DATA_LOC = DATA_LOCS[loc]
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
                ret,thresh = cv2.threshold(mask2,127,255,0)

                ''' kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                thresh = cv2.erode(thresh, kernel)
                thresh = cv2.dilate(thresh, kernel) '''

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

                res = contours[ci]
                hull = cv2.convexHull(res)
                #final_image = np.zeros(image.shape, np.uint8)
                #cv2.drawContours(final_image, [res], 0, (0, 255, 0), 2)
                #cv2.drawContours(final_image, [hull], 0, (0, 0, 255), 3)

                # Create a ROI
                x1,y1,w1,h1 = cv2.boundingRect(res)
                #cv2.rectangle(final_image, (x1,y1), (x1+w1,y1+h1), (255,0,0), thickness=2)

                (x,y),(major_axis,minor_axis),angle = cv2.fitEllipse(res)
                #cv2.ellipse(final_image,(int(x),int(y)),(int(major_axis/2),int(minor_axis/2)),angle,0.0,360.0,(0,255,0),1)

                center = (x-x1,y-y1)
                eccentricity = (1 - (major_axis/minor_axis) ** 2 ) ** 0.5
                contour_area = cv2.contourArea(res)
                scale = contour_area/(w1*h1)
                act_r = ((center[0])**2+(center[1])**2)**0.5
                angle = 180-angle if angle > 90 else angle
                dump_file.write('%.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,,,,%c\n' % (angle/90, act_r, 3.142*major_axis*minor_axis*0.25, contour_area, scale, eccentricity, label))
                total_images_parsed += 1
            except Exception as e:
                #print(e)
                break
total = (time.time() - start)
print(' '*160+'\rTotal time required = %3.3fs' % (total))
print('Total images parsed: %d'%(total_images_parsed))
winsound.Beep(1000, 100)
winsound.Beep(1200, 100)
winsound.Beep(1500, 100)
winsound.Beep(1700, 100)
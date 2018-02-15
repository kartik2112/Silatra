import cv2, numpy as np, time, math
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

f = open('bounds.txt')
param = int(f.read().strip())
lower = np.array([0,137,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

start = time.time()
dump_file = open('testing_data_letters.csv','w')
''' data = pd.read_csv('train_img_cha_data.csv')

X = data[['norm_r', 'scale', 'eccentricity']]
Y = data['label']

r_data = data['r'].values.tolist()
MIN_VAL, MAX_VAL = min(r_data), max(r_data)

print('MAX: %3.3f, MIN: %3.3f' % (MAX_VAL, MIN_VAL))

X_train, X_test, Y_train, Y_test = tts(X,Y,test_size=0.3,random_state=137)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, Y)

predictions = [ [ 0 for i in range(10)] for i in range(10)]
accuracy, total = 0, 0 '''
dump_file.write('angle_90,r,area_ellipse,area_contour,scale,eccentricity,norm_r,norm_area_ellipse,norm_area_contour,label\n')
data_loc_changed = False
for label in ['a','b','c','d','e']:
    #if data_loc_changed: DATA_LOC = 'T:\\Tejas\\Coding Workspace\\Untracked\\training-images-test\\Digits_Kartik\\'
    DATA_LOC = 'training-images-varun\\Letters\\'
    for i in range(1,1000):
        try:
            print(' '*160+'\rProcessing image: %3d, Label = %c, From Location: %s' % (i,label,DATA_LOC),end='\r')
            image = cv2.imread(DATA_LOC+label+'\\'+str(i)+'.png')

            HEIGHT, WIDTH, _ = image.shape

            blur = cv2.blur(image,(3,3))
            ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)

            #Create a binary image with where white will be skin colors and rest is black
            mask2 = cv2.inRange(ycrcb,lower,upper)
            ret,thresh = cv2.threshold(mask2,127,255,0)

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
            x1,y1,w1,h1 = cv2.boundingRect(hull)
            #cv2.rectangle(final_image, (x1,y1), (x1+w1,y1+h1), (255,0,0), thickness=2)

            (x,y),(major_axis,minor_axis),angle = cv2.fitEllipse(hull)
            #cv2.ellipse(final_image,(int(x),int(y)),(int(major_axis/2),int(minor_axis/2)),angle,0.0,360.0,(0,255,0),1)

            center = (x-x1,y-y1)
            eccentricity = (1 - (major_axis/minor_axis) ** 2 ) ** 0.5
            hull_area = cv2.contourArea(hull)
            scale = hull_area/(WIDTH*HEIGHT)
            act_r = ((center[0])**2+(center[1])**2)**0.5
            angle = 180-angle if angle > 90 else angle
            ''' norm_r = (act_r-MIN_VAL)/(MAX_VAL-MIN_VAL)
            prediction = classifier.predict([[norm_r, scale, eccentricity]])[0]
            predictions[label][prediction] += 1
            if prediction == label: accuracy += 1
            total += 1 '''
            dump_file.write('%.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,,,,%c\n' % (angle/180, act_r, 3.142*major_axis*minor_axis*0.25, hull_area, scale, eccentricity, label))
        except Exception as e:
            #print(e)
            break
    #data_loc_changed = not data_loc_changed
total = (time.time() - start)
print(' '*160+'\rTotal time required = %3.3fs' % (total))
''' for row_prediction in predictions:
    print('[',end=' ')
    for prediction in row_prediction:
        print('%3d'%(prediction),end=' ')
    print(']')
print('Total images processed = %4d'%(total))
print('Accuracy = %3.3f'%(accuracy/total*100)) '''
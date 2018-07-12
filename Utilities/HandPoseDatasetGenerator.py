# This program is used to generate hand poses dataset.
# This program is for generating dataset of digits.
# Manipulation of this program is needed to capture letters and gesture signs.
# When you execute this program, you first need to enter the digit whose dataset you are trying to create.
# Then, you need to keep holding key 'c' so as to capture all data.
# If you click key 'q', the program will exit.


import cv2, numpy as np, argparse
import silatra
import os

ap = argparse.ArgumentParser()
ap.add_argument('-s','--sign',help='This is the label and the folder name where your images would be stored')
args = vars(ap.parse_args())

lower = np.array([0,140,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

cap = cv2.VideoCapture(0)
#cap.set(3,640); cap.set(4,480)
#cap.set(cv2.CAP_PROP_FPS, 5)

total_captured = 390
if args.get('sign'): sign = args.get('sign')
else: sign = int(input('Ab konsa no: '))

if not os.path.exists("../Dataset/TempFolder/%s/"%(sign)):
    os.makedirs("../Dataset/TempFolder/%s/"%(sign))
while True:
    _, frame = cap.read()
    x,y,w,h = 70,50,300,300

    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), thickness=2)
    cv2.putText(frame, '%d'%(total_captured), (550,450), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), thickness=3)
    roi = frame[y:y+h,x:x+w]
    # ycrcb = cv2.cvtColor(roi,cv2.COLOR_BGR2YCR_CB)
    # mask = cv2.inRange(ycrcb,lower,upper)
    mask, foundFace, faceRect = silatra.segment(roi)
    #skin = cv2.bitwise_and(roi, roi, mask = mask)
    _,thresh = cv2.threshold(mask,127,255,0)
    ''' kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.erode(thresh,kernel) '''
    
    cv2.imshow('You',frame)
    cv2.imshow('Hand',thresh)
    k=cv2.waitKey(20)
    print('Total captured: %3d' % (total_captured),end='\r')
    if k==ord('q'): break
    elif k==ord('c'):
        if total_captured < 500:
            cv2.imwrite('../Dataset/TempFolder/%s/%d.png'%(sign,total_captured+1),roi)
            #cv2.imwrite('test.png',roi)
            total_captured += 1
        else: break
cv2.destroyAllWindows()
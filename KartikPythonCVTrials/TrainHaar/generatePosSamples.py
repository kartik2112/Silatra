import os
import cv2
import numpy as np

folders = ['training-images/Digits/1/Right_Hand/Normal',
            'training-images/Digits/2/Right_Hand/Normal',
            'training-images/Digits/3/Right_Hand/Normal',
            'training-images/Digits/4/Right_Hand/Normal',
            'training-images/Digits/5/Right_Hand/Normal',
            'training-images/Digits/6/Right_Hand/Normal',
            'training-images/Digits/7/Right_Hand/Normal',
            'training-images/Digits/8/Right_Hand/Normal',
            'training-images/Digits/9/Right_Hand/Normal',
            'training-images/Digits/0/Right_Hand/Normal']

imgNo = 1
for folderPath in folders:
    
    # if not os.path.exists("./pos/"+folderPath+"/"):
    #     os.makedirs("./pos/"+folderPath+"/")
    ctr = 120
    for img1 in os.listdir("../../"+folderPath):
        img = cv2.imread("../../"+folderPath+"/"+img1)
        img = cv2.GaussianBlur(img,(5,5),0)
        img = cv2.medianBlur(img,5)

        imgYCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)

        mask1 = cv2.inRange(imgYCrCb,np.array([0,137,100],np.uint8),np.array([255,200,150],np.uint8))

        morphOpenElem = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
        morphCloseElem = cv2.getStructuringElement(cv2.MORPH_CROSS,(13,13))

        mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,morphOpenElem)
        mask1 = cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,morphCloseElem)

        dilateElem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask1 = cv2.dilate(mask1,dilateElem,3)

        im2, contours, hierarchy = cv2.findContours( mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        maxAr = 0
        maxArCntInd = -1
        # cv2.drawContours(img,contours,-1,(255,0,0),1)
        # print(contours)
        for i in range(len(contours)):
            hull1 = cv2.convexHull(contours[i])
            ar1 = cv2.contourArea(hull1)
            # print(ar1,maxAr)
            if ar1>maxAr:
                maxAr = ar1
                maxArCntInd = i
                # print(i)
        # cv2.drawContours(img,[contours[maxArCntInd]],0,(0,255,0),1)

        (x,y,w,h) = cv2.boundingRect(contours[maxArCntInd])
        x-=10
        y-=10
        w+=20
        h+=20

        if w>h:
            y-=int((w-h)/2)
            h=w
        else:
            x-=int((h-w)/2)
            w=h
        iH,iW,iC = img.shape

        if x<0:
            x=0
        if y<0:
            y=0

        if y+h>iH:
            y-=(y+h-iH+1)
            print("H exceeds!!")
        if x+w>iW:
            x-=(x+w-iW+1)
            print("W exceeds!!")

        # print(rect)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        # print(img1)
        roi = img[y:y+h,x:x+w]
        roi = cv2.resize(roi,(60,60))
        
        # cv2.imshow("Img1",img)
        # cv2.imshow("Img",roi)
        cv2.imwrite("./info/"+str(imgNo)+".png",roi)
        print(folderPath+"/"+img1)
        imgNo+=1
        ctr-=1
        if ctr == 0:
            break
        # if cv2.waitKey(200) == ord('q'):
        #     break
    # if cv2.waitKey(20) == ord('q'):
    #     break
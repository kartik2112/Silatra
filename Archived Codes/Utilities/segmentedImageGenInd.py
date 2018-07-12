import cv2
import silatra
import os.path

dir = "../training-images/Letters/"
for sign123 in range(ord('a'),ord('z')+1):
    sign1 = chr(sign123)
    try:
        if not(os.path.isfile(dir + str(sign1) + "/20.png")):
            continue
        img1 = cv2.imread(dir + str(sign1) + "/20.png")
    except:
        continue
    mask1,_,_ = silatra.segment(img1)


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
    if maxArCntInd == -1:
        print("Couldnt find contour")
        continue
        
    (x,y,w,h) = cv2.boundingRect(contours[maxArCntInd])
    
    mask1 = cv2.cvtColor(mask1,cv2.COLOR_GRAY2BGR)

    imgFin = cv2.bitwise_and(mask1,img1)

    roi = imgFin[y:y+h,x:x+w]



    cv2.imwrite("PosterImages/Letters/"+str(sign1)+".png",roi)
    print("Working on this 1")

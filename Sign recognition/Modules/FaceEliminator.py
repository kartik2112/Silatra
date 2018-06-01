import numpy as np
import cv2
import imutils

MIN_AREA_THRESHOLD = 300

def eliminateFace(mask,foundFace,face):
    global MIN_AREA_THRESHOLD
    # Connect face to neck using rectangle by following anthropometry
    
    HEIGHT, WIDTH = mask.shape
    if foundFace:
        (x,y,w,h) = face
        # cv2.rectangle(mask, (x, y), (x + w, y + h), (0,0,0), -1)
        faceNeckExtraRect = ((int(x+(w/2)-8), int(y+h/2)), (int(x+(w/2)+8), int(y+h+h/4)))
        cv2.rectangle(mask, faceNeckExtraRect[0], faceNeckExtraRect[1], (255,255,255), -1)
        
        tempImg1 = np.zeros((HEIGHT,WIDTH,1), np.uint8)
        cv2.rectangle(tempImg1, (x, y), (x + w, y + h), (0,0,0), -1)
        cv2.rectangle(tempImg1, faceNeckExtraRect[0], faceNeckExtraRect[1], (255,255,255), -1)
    
    _,contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    max_area_of_intersection = -1
    intersectingContour = -1
    # cv2.drawContours(mask, contours, -1, (0,255,0), 2)
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area < MIN_AREA_THRESHOLD:
                cv2.drawContours(mask, contours, i, (0,0,0), -1)
                continue
            if foundFace:                
                tempImg2 = np.zeros((HEIGHT,WIDTH,1), np.uint8)
                cv2.rectangle(tempImg1, (x, y), (x + w, y + h), (255,255,255), -1)
                cv2.drawContours(tempImg2, contours, i, (255,255,255), -1)
                tempImg3 = cv2.bitwise_and(tempImg1,tempImg2)
                area_of_intersection = np.sum(tempImg3 == 255)
                if area_of_intersection > max_area_of_intersection:
                    max_area_of_intersection = area_of_intersection
                    intersectingContour = i
        if intersectingContour != -1:
            cv2.drawContours(mask, contours, intersectingContour, (0,0,0), -1)
            # cv2.rectangle(mask, faceNeckExtraRect[0], faceNeckExtraRect[1], (255,255,255), 2)
            # cv2.rectangle(mask, (x, y), (x + w, y + h), (255,255,255), 2)
    return mask

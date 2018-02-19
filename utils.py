import cv2, numpy as np
from math import ceil

def segment(src_img, lower_bounds, upper_bounds):
    """
    ### Segment skin areas from hand using a YCrCb mask.

    This function returns a mask with white areas signifying skin and black areas otherwise.
    Skin area is defined between `lower_bounds` and `upper_bounds`.

    Returns: mask
    """
    blur = cv2.blur(src_img,(3,3))
    ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)
    mask = cv2.inRange(ycrcb,lower_bounds,upper_bounds)
    return mask

def get_my_hand(img_gray,return_only_contour=False):
    """
    ### Hand extractor

    __DO NOT INCLUDE YOUR FACE IN THE `img_gray`__
    
    Provide an image where skin areas are represented by white and black otherwise.
    This function does the hardwork of finding your hand area in the image.

    Returns: *(image)* Your hand conoturs.
    """
    _,contours,_ = cv2.findContours(img_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
    if return_only_contour: return contours[ci]
    else:
        x,y,w,h = cv2.boundingRect(contours[ci])
        hand = np.zeros((img_gray.shape[1], img_gray.shape[0], 1), np.uint8)
        cv2.drawContours(hand, contours, ci, 255, cv2.FILLED)
        _,hand = cv2.threshold(hand[y:y+h,x:x+w], 127,255,0)
        return hand
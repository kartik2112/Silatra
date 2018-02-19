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

    @return:
    return_only_contour == True then contour of your hand
    else returns hand as an image.
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

def extract_features(src_hand, grid):
    """
    ### Feature extractor

    Provide image of hand only to this function. The Grid size (x,y) is also required.

    @return Array of features
    """
    HEIGHT, WIDTH = src_hand.shape

    data = [ [0 for haha in range(grid[0])] for hah in range(grid[1]) ]
    h, w = float(HEIGHT/grid[1]), float(WIDTH/grid[0])
    
    for column in range(1,grid[1]+1):
        for row in range(1,grid[0]+1):
            fragment = src_hand[ceil((column-1)*h):min(ceil(column*h), HEIGHT),ceil((row-1)*w):min(ceil(row*w),WIDTH)]
            _,contour,_ = cv2.findContours(fragment,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            try: area = cv2.contourArea(contour[0])
            except: area=0.0
            area = float(area/(h*w))
            data[column-1][row-1] = area
    
    features = []
    for column in range(grid[1]):
        for row in range(grid[0]):
            features.append(data[column][row])
    return features
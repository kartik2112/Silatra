import cv2, numpy as np
from math import ceil
import silatra

def segment(src_img, lower_bounds, upper_bounds):
    """
    ### Segment skin areas from hand using a YCrCb mask.

    This function returns a mask with white areas signifying skin and black areas otherwise.
    Skin area is defined between `lower_bounds` and `upper_bounds`.

    Returns: mask
    """
    # blur = cv2.blur(src_img,(3,3))
    # ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)
    # mask = cv2.inRange(ycrcb,lower_bounds,upper_bounds)
    mask = silatra.segment(src_img)
    return mask

def get_my_hand(img_gray, return_only_contour=False):
    """
    ### Hand extractor

    __DO NOT INCLUDE YOUR FACE IN THE `img_gray`__
    
    Provide an image where skin areas are represented by white and black otherwise.
    This function does the hardwork of finding your hand area in the image.

    Returns: *(image)* Your hand.
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
        # hand = np.zeros((img_gray.shape[1], img_gray.shape[0], 1), np.uint8)
        # cv2.drawContours(hand, contours, ci, 255, cv2.FILLED)
        # _,hand = cv2.threshold(hand[y:y+h,x:x+w], 127,255,0)
        hand = img_gray[y:y+h,x:x+w]
        return hand

def extract_features(src_hand, grid):
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



def addToQueue(pred):
    '''
    Adds the latest sign recognized to a queue of signs. This queue has maxlength: `maxQueueSize`

    Parameters
    ----------
    pred : This is the latest sign recognized by the classifier.
            This is of type number and the sign is in ASCII format

    '''
    global preds, maxQueueSize, minModality, noOfSigns
    print("Received Sign:",pred)
    if len(preds) == maxQueueSize:
        preds = preds[1:]
    preds += [pred]
    

def getConsistentSign():
    '''
    From the queue of signs, this function returns the sign that has occured most frequently 
    with frequency > `minModality`. This is considered as the consistent sign.

    Returns
    -------
    number
        This is the modal value among the queue of signs.

    '''
    global preds, maxQueueSize, minModality, noOfSigns
    modePrediction = -1
    countModality = minModality

    if len(preds) == maxQueueSize:
        countPredictions = [0]*noOfSigns

        for pred in preds:
            if pred != -1:
                countPredictions[pred]+=1
        
        for i in range(noOfSigns):
            if countPredictions[i]>countModality:
                modePrediction = i
                countModality = countPredictions[i]

        displaySignOnImage(modePrediction)
    
    return modePrediction

def displayTextOnWindow(windowName,textToDisplay):
    '''
    This just displays the text provided on the cv2 window with WINDOW_NAME: `windowName`

    Parameters
    ----------
    windowName : This is WINDOW_NAME of the cv2 window on which the text will be displayed
    textToDisplay : This is the text to be displayed on the cv2 window

    '''
    signImage = np.zeros((200,200,1),np.uint8)

    cv2.putText(signImage,textToDisplay,(75,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,8);

    cv2.imshow(windowName,signImage);

def displaySignOnImage(predictSign):
    '''
    This abstracts the logic for handling signs that have not been detected in majority.

    Parameters
    ----------
    predictSign : This is the recognized sign (in ASCII) to be displayed on the cv2 window

    '''
    dispSign = "--"
    if predictSign != -1:
        dispSign = chr(predictSign)+"";

    displayTextOnWindow("Prediction",dispSign)


def recordTimings(start_time,time_key):
    '''
    This performs the manipulation of average, min, max timings for each of the components

    Parameters
    ----------
    start_time : This is the base reference start_time:Timer with reference to which the current time is measured
                and the difference is the time elapsed which is used for calculation of average, min, max timings
    time_key : This indicates the timings of which component need to be updated.
    
    Returns
    -------
    Timer
        This function returns the current instance of timer so that, this can be used in the next invokation of this function.

    '''
    global minTimes,maxTimes,avgTimes,noOfFramesCollected
    if noOfFramesCollected != 0: 
        elapsed = timeit.default_timer() - start_time
        avgTimes[time_key] = avgTimes[time_key] * ((noOfFramesCollected-1)/noOfFramesCollected) + elapsed/noOfFramesCollected
        minTimes[time_key] = elapsed if elapsed < minTimes[time_key] else minTimes[time_key]
        maxTimes[time_key] = elapsed if elapsed > maxTimes[time_key] else maxTimes[time_key]
    return timeit.default_timer()


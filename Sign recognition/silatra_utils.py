import cv2, numpy as np
from math import ceil
import silatra



'''
* These variables form part of the logic that is used for stabilizing the stream of signs
* From a stream of most recent `maxQueueSize` signs, the sign that has occured most frequently 
*   with frequency > `minModality` is considered as the consistent sign
'''
preds = []          # This is used as queue for keeping track of last `maxQueueSize` signs for finding out the consistent sign
maxQueueSize = 15   # This is the max size of queue `preds`
noOfSigns = 128     # This is the domain of the values present in the queue `preds`
minModality = int(maxQueueSize/2)   # This is the minimum number of times a sign must be present in `preds` to be declared as consistent



def get_my_hand(image_skin_mask):
    """
    ### Hand extractor

    __DO NOT INCLUDE YOUR FACE IN THE `image_skin_mask`__
    
    Provide an image where skin areas are represented by white and black otherwise.
    This function does the hardwork of finding your hand area in the image.

    Returns: *(image)* Your hand, *(hand_contour)*.
    """
    _,contours,_ = cv2.findContours(image_skin_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    ci = -1
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
    if ci == -1:
        return [ False, None, None  ]
    x,y,w,h = cv2.boundingRect(contours[ci])
    # hand = np.zeros((image_skin_mask.shape[1], image_skin_mask.shape[0], 1), np.uint8)
    # cv2.drawContours(hand, contours, ci, 255, cv2.FILLED)
    # _,hand = cv2.threshold(hand[y:y+h,x:x+w], 127,255,0)
    hand = image_skin_mask[y:y+h,x:x+w]

    return [ True, hand, contours[ci] ]




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



def predictSign(classifier,features):
    predictions = classifier.predict_proba([features]).tolist()[0]
    # print(classifier.predict_proba([features]).tolist())
    for prob in predictions: print('%.2f'%(prob),end=' ')
    print('')

    pred = classifier.predict([features])[0] 
    print(pred)
    return pred








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

def displayTextOnWindow(windowName,textToDisplay,xOff=75,yOff=100):
    '''
    This just displays the text provided on the cv2 window with WINDOW_NAME: `windowName`

    Parameters
    ----------
    windowName : This is WINDOW_NAME of the cv2 window on which the text will be displayed
    textToDisplay : This is the text to be displayed on the cv2 window

    '''
    signImage = np.zeros((200,200,1),np.uint8)

    cv2.putText(signImage,textToDisplay,(xOff,yOff),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,8);

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



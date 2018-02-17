# Reference: https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/

from skimage.measure import compare_ssim
import numpy as np
import cv2

cam = cv2.VideoCapture(0)
(ret,framePrev) = cam.read()

# framePrev = cv2.GaussianBlur(framePrev,(5,5),0)
# imgYCrCb = cv2.cvtColor(framePrev,cv2.COLOR_BGR2YCR_CB)

# mask1 = cv2.inRange(imgYCrCb,np.array([0,137,100],np.uint8),np.array([255,200,150],np.uint8))

# morphOpenElem = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
# morphCloseElem = cv2.getStructuringElement(cv2.MORPH_CROSS,(13,13))

# mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,morphOpenElem)
# mask1 = cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,morphCloseElem)

# dilateElem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# maskPrev = cv2.dilate(mask1,dilateElem,3)
# maskPrev = cv2.GaussianBlur(maskPrev,(9,9),0)

framePrev = cv2.cvtColor(framePrev, cv2.COLOR_BGR2GRAY)
framePrev = cv2.GaussianBlur(framePrev,(3,3),0)

while True:
    (ret,frame) = cam.read()
    # frame = cv2.GaussianBlur(frame,(5,5),0)
    # imgYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)

    # mask1 = cv2.inRange(imgYCrCb,np.array([0,137,100],np.uint8),np.array([255,200,150],np.uint8))

    # morphOpenElem = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    # morphCloseElem = cv2.getStructuringElement(cv2.MORPH_CROSS,(13,13))

    # mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,morphOpenElem)
    # mask1 = cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,morphCloseElem)

    # dilateElem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # mask1 = cv2.dilate(mask1,dilateElem,3)
    # mask1 = cv2.GaussianBlur(mask1,(9,9),0)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(3,3),0)

    # (score,diff) = compare_ssim(mask1,maskPrev, full=True)
    (score,diff) = compare_ssim(frame,framePrev, full=True)
    # print("Similarity:",(1-(score+1)/2))
    print("Similarity:",score)

    if(score<0.88):
        print("Stop checking for signs")

    cv2.imshow("Image",frame)
    # cv2.imshow("Mask",mask1)
    cv2.imshow("Diff",diff)

    framePrev = frame
    # maskPrev = mask1
    if cv2.waitKey(10)==ord('q'):
        break
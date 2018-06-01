import numpy as np
import cv2
import imutils


'''
* For stabilization of person, KCF Tracker is used. Now for this face detected using Haar is used as ROI.
* Sometimes the KCF Tracker fails to locate the ROI. For this purpose, even after waiting for `maxNoOfFramesNotTracked` frames,
*   if the tracker fails to locate the ROI, the tracker is reinitialized.
'''
faceStabilizerMode = "OFF"  # This is used to enable/disable the stabilizer using KCF Tracker
trackingStarted = False     # This is used to indicate whether tracking has started or not
noOfFramesNotTracked = 0    # This indicates the no of frames that has not been tracked
maxNoOfFramesNotTracked = 15 # This is the max no of frames that if not tracked, will restart the tracker algo
minNoOfFramesBeforeStabilizationStart = 50
trackerInitFace = (0,0,0,0)
tracker = cv2.TrackerKCF_create()



def stabilize(foundFace,noOfFramesCollected,img_np,faceRect,mask1):
    '''
    * Here is the stabilization logic
    *
    * We are stabilizing the person by using face as the ROI for tracker. Thus, in situations where
    * the person moves while the camera records the frames, or if the camera operator's hand shakes, 
    * these false movements wont be detected.
    * We are using `noOfFramesCollected` so as to improve the stabilization results by delaying the
    * tracker initialization
    '''
    global trackingStarted, minNoOfFramesBeforeStabilizationStart, tracker, trackerInitFace, noOfFramesNotTracked, maxNoOfFramesNotTracked
    if not(trackingStarted) and foundFace and noOfFramesCollected >= minNoOfFramesBeforeStabilizationStart:
        trackingStarted = True
        ok = tracker.init(img_np, faceRect)
        trackerInitFace = faceRect
    elif trackingStarted:
        ok, bbox = tracker.update(img_np)
        if ok:
            cv2.rectangle(img_np, (int(bbox[0]),int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (255,0,0), 2)
            
            # img_np1 = img_np.copy()
            # cv2.putText(img_np1,str("("+str(int(bbox[0]))+","+str(int(bbox[1]))+")"),(int(bbox[0]-10),int(bbox[1]-20)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,8);
            # cv2.imshow("OG Img",img_np1)
            cv2.imshow("OG Img",img_np)
            # cv2.imwrite('../training-images/kartik/SampleImages/%d_OG.png'%(total_captured),img_np1)
            rows,cols,_ = img_np.shape
            tx = int(trackerInitFace[0] - bbox[0])
            ty = int(trackerInitFace[1] - bbox[1])
            shiftMatrix = np.float32([[1,0,tx],[0,1,ty]])
            
            # Reference: https://www.docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
            img_np = cv2.warpAffine(img_np,shiftMatrix,(cols,rows))
            mask1 = cv2.warpAffine(mask1,shiftMatrix,(cols,rows))

            # cv2.putText(img_np,str("("+str(trackerInitFace[0])+","+str(trackerInitFace[1])+")"),(int(trackerInitFace[0]-10),int(trackerInitFace[1]-20)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,8);
            cv2.imshow("Stabilized Image",img_np)
            # cv2.imwrite('../training-images/kartik/SampleImages/%d_Stabilized.png'%(total_captured),img_np)
            noOfFramesNotTracked = 0
            # cv2.imshow("Stabilized Mask",mask1)
        else:
            noOfFramesNotTracked += 1
            if noOfFramesNotTracked > maxNoOfFramesNotTracked:
                trackingStarted = False
                noOfFramesNotTracked = 0
# FeatureDetectionSIFT.py for trying out OpenCV Feature Detection using SIFT

import cv2
import numpy as np
img = cv2.imread('training-images/Digits/3/Right_Hand/Normal/2.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp,img)
cv2.imwrite('sift_keypoints.jpg',img)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)

print("Size of numpy array:"+str(des.shape))
import matplotlib.pyplot as plt
plt.hist(des, bins='auto')
plt.show()

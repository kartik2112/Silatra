import cv2
import numpy as np
import os

pathRoot = "phone"

images = os.listdir(pathRoot)
for imageName in images:
    print("Resizing "+pathRoot+"/"+imageName)
    img = cv2.imread(pathRoot+"/"+imageName)
    img1 = cv2.resize(img,(0,0),fx=0.4,fy=0.4)
    cv2.imwrite(pathRoot+"/"+imageName,img1)
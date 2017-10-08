# FeatureDetectionSIFT.py for trying out OpenCV Feature Detection using SIFT
vocab_size=3
import cv2
import numpy as np
from sklearn.cluster import KMeans
OutputFile=open("totalData.csv","w")
header=""
for i in range(0,vocab_size):
    header+="F"+str(i)+","
header+="target\n"
OutputFile.write(header)
for digit in range(1,6):
    print("Reading from path:"+"training-images/Digits/"+str(digit)+"/Right_Hand/Normal/")
    for named_num in range(1,35):
        path='training-images/Digits/'+str(digit)+'/Right_Hand/Normal/'+str(named_num)+'.png'
        img = cv2.imread(path)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray,None)
        img=cv2.drawKeypoints(gray,kp,img)
        cv2.imwrite('sift_keypoints.jpg',img)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)
        # print(des)
        kmeans = KMeans(n_clusters=vocab_size, random_state=0).fit(des)
        histoBins=[0]*vocab_size
        # print(kmeans.labels_)
        for label in kmeans.labels_:
            histoBins[label]+=1
        CSVRecord=''
        for Bin in histoBins:
            CSVRecord+=str(Bin)
            CSVRecord+=','
        CSVRecord+=str(digit)+'\n'
        OutputFile.write(CSVRecord)
print("CSV File prepared.")
OutputFile.close()

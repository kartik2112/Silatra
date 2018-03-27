import cv2
import numpy as np
import os
import argparse
import imutils

# parser = argparse.ArgumentParser(description='This is just used to save the video frames as seperate images on disk')
# parser.add_argument('videoURI',help='Relative path of video to be converted into frames')
# parser.add_argument('outputDir',help='Relative path of output folder where frames are to be stored')

# args = parser.parse_args()
# videoURI = args.videoURI
# outputDir = args.outputDir

videoURIs = ['/media/kartik/KARTIK/HandDataset/videos/20180326_175745.mp4']
outputDir = "/media/kartik/KARTIK/HandDataset/videos"

for videoURI in videoURIs:
    # print(videoURI)
    # print(outputDir)
    videoName = videoURI.split("/")[-1].split(".")[0]
    # print(videoName)

    if not os.path.isfile(videoURI):
        print("Specified File does not exist")
    else:
        cap = cv2.VideoCapture(videoURI)
        frameNo = 1
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            frame = imutils.rotate_bound(frame,90)
            print("Processing frame: %d"%(frameNo)+" and writing at "+outputDir+"/"+videoName+"_%06d.png"%(frameNo),end="\r")
            cv2.imwrite(outputDir+"/"+videoName+"_%06d.png"%(frameNo),frame)
            frameNo+=1
        print("Written %d frames to %s using video:%s      "%(frameNo,outputDir,videoURI))

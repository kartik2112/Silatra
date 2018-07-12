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

videoURIs = [
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_001.avi',
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_002.avi',
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_003.avi',
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_004.avi',
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_005.avi',
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_006.avi',
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_007.avi',
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_008.avi',
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_009.avi',
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_010.avi',
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_011.avi',
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_012.avi',
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_013.avi',
'/home/kartik/Documents/Projects/BE_Project/Silatra/training-images/GestureVideos/ObjTrainingVids/Gesture_ObjTrainingVids_014.avi']


outputDir = "/media/kartik/0EAB-5DFB/HandSignsDataset/AllFrames"

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
            # frame = imutils.rotate_bound(frame,90)
            print("Processing frame: %d"%(frameNo)+" and writing at "+outputDir+"/"+videoName+"_%06d.png"%(frameNo),end="\r")
            cv2.imwrite(outputDir+"/"+videoName+"_%06d.png"%(frameNo),frame)
            frameNo+=1
        print("Written %d frames to %s using video:%s      "%(frameNo,outputDir,videoURI))

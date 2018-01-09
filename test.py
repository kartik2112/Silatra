import cv2
from skin_detector import segment
from numpy import hstack

# import the necessary packages
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help = "path to the (optional) video file")
args = vars(ap.parse_args())
 
# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0,140,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
 
# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])


start_tracking = False
# keep looping over the frames in the video
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    skin = segment(frame)

    r,h,c,w = 300,150,300,150

    frame = skin.copy()
    skin = cv2.rectangle(skin, (r,c), (r+w,c+h), (0,0,255), 2)

    skin = cv2.resize(skin,(640,320))
    cv2.imshow('segmented',skin)

    ''' if not start_tracking:
        tracker = cv2.Tracker_create('MIL')
        bounding_box = (r, c, w, h)
        #bounding_box = cv2.selectROI(frame, False)
        ok = tracker.init(frame, bounding_box)
        cv2.imshow('Place your hand within the red box',skin)
    else:
        ok = tracker.init(frame, bounding_box)
        ok, bounding_box = tracker.update(frame)
        if ok:
            # Tracking is successful
            p1 = (int(bounding_box[0]), int(bounding_box[1]))
            p2 = (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        cv2.imshow("Tracking result", frame) '''
    # if the 'q' key is pressed, stop the loop
    k=cv2.waitKey(1)
    if k == ord("q"):
        break
    elif k == ord('s'):
        start_tracking = True

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
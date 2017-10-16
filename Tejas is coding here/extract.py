# import the necessary packages
import imutils
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
lower = np.array([0,143,77],np.uint8)
upper = np.array([255,173,127],np.uint8)

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
 
# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping over the frames in the video
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the speicifed upper and lower boundaries
    frame = imutils.resize(frame, width = 400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.dilate(skinMask, kernel, iterations = 3)
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    """
    image = skin.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh, 2,1)
    cnt = cnts[0]
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt,hull)

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(image,start,end,[0,255,0],2)
        cv2.circle(image,far,5,[0,0,255],-1)
    
    """

    # ORB feature detection code
    image = skin.copy()
    orb = cv2.ORB_create()
    kp = orb.detect(image, None)
    kp,des = orb.compute(image,kp)

    # Get keypoints
    img = cv2.drawKeypoints(image, kp, None, color=(0,255,0), flags=0)

    cv2.imshow("Actual image, Skin detected, Extreme points detected", np.hstack([frame, skin, img]))
    
    # if the 'q' key is pressed, stop the loop
    k=cv2.waitKey(1)
    if k == ord("q"):
        break
    elif k == ord("c"):
        with open("Data/1.csv","a") as f:
            s=''
            for keyp in kp:
               p=keyp.pt
               f.write(str(p[0])+","+str(p[1]))
            f.write('\n')


# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
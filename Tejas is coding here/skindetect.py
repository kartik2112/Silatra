''' # import the necessary packages
import imutils
import numpy as np
import argparse
import cv2
 
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
	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
	skinMask = cv2.erode(skinMask, kernel, iterations = 2)
 
	# blur the mask to help remove noise, then apply the
	# mask to the frame
	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	skin = cv2.bitwise_and(frame, frame, mask = skinMask)
 
	# show the skin in the image along with the mask
	cv2.imshow("images", np.hstack([frame, skin]))
 
	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		cv2.imwrite('img.png',skin)
		break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows() '''

import cv2
import numpy as np

image = cv2.imread('../SkinDetection/Test_Images/test_img.jpg').tolist()
#image = cv2.imread('../training-images/Digits/1/Right_Hand/Normal/1.png').tolist()
threshold = -0.0

for i in range(len(image)):
	for j in range(len(image[i])):
		pixel = image[i][j]
		if pixel[0]-pixel[2] < threshold and pixel[1]-pixel[2] < threshold: # B<G<R or R>G>B
			image[i][j][0] = image[i][j][1] = image[i][j][2] = 255.0
		else: image[i][j] = [0,0,0]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
skinMask = cv2.dilate(np.array(image), kernel, iterations = 4)
cv2.imshow('After condition',np.array(image))
cv2.waitKey(10000)
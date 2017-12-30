import cv2
from skin_detector import predict_skin_pixels
from numpy import hstack

img_file = input('Image: ')
only_skin_img = predict_skin_pixels(img_file)
cv2.imshow('Result', hstack([only_skin_img]))
cv2.waitKey(100000)
cv2.destroyAllWindows()
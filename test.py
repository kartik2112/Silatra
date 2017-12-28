import cv2
from skin_detector import predict_skin_pixels

img_file = input('Image: ')
only_skin_img = predict_skin_pixels(img_file)
cv2.imshow('Segmented image', only_skin_img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
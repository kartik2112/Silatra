import numpy as np
import cv2

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

hand_cascade = cv2.CascadeClassifier('hand_cascade_SK.xml')
cam = cv2.VideoCapture(0)
while(True):
    (ret,img) = cam.read()
    # img = cv2.imread('sachin.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = hand_cascade.detectMultiScale(gray, 2.1, 20)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',img)
    if cv2.waitKey(10)==ord('q'):
        break
cv2.destroyAllWindows()
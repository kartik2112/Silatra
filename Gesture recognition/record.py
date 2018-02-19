import cv2, numpy as np
from utils import segment

lower = np.array([0,147,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('test2.avi',fourcc, 10, (300,300))

start_rec = False
while(1):
    _,frame = cap.read()
    mask = segment(frame, lower, upper)

    x,y,w,h = 100,100,300,300
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    roi = frame[y:y+h,x:x+w]

    cv2.imshow('You', frame)
    if not start_rec: cv2.imshow('Segmented', mask)
    else: out.write(roi)
    
    k=cv2.waitKey(50)
    if k==ord('q'): break
    elif k==ord('s'):
        start_rec = not start_rec
        cv2.destroyAllWindows()
cap.release()
out.release()
cv2.destroyAllWindows()
import cv2, numpy as np

lower = np.array([0,140,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3,640); cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS, 5)

total_captured = 0
sign = 'c'
while True:
    _, frame = cap.read()
    x,y,w,h = 300,150,300,300

    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), thickness=2)
    cv2.putText(frame, '%d'%(total_captured), (50,150), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), thickness=3)
    roi = frame[y:y+h,x:x+w]
    ycrcb = cv2.cvtColor(roi,cv2.COLOR_BGR2YCR_CB)
    mask = cv2.inRange(ycrcb,lower,upper)
    skin = cv2.bitwise_and(roi, roi, mask = mask)
    _,thresh = cv2.threshold(mask,127,255,0)
    
    cv2.imshow('You',frame)
    cv2.imshow('Hand',thresh)
    k=cv2.waitKey(50)
    print('Total captured: %3d' % (total_captured),end='\r')
    if k==ord('q'): break
    elif k==ord('c'):
        if total_captured < 300:
            cv2.imwrite('training-images-varun/Letters/%c/%d.png' % (sign,total_captured+1),roi)
            total_captured += 1
cv2.destroyAllWindows()
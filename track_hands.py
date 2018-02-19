import cv2, numpy as np, time, math
from utils import segment, get_my_hand

#Open Camera object
cap = cv2.VideoCapture('test.avi')
cap.set(3,640); cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS, 20)

contour_start=False
f = open('bounds.txt')
param = int(f.read().strip())
f.close()
lower = np.array([0,param,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

prev_x, prev_y = 0, 0
THRESHOLD = 20
while(1):
    try:
        start_time = time.time()
        ret, frame = cap.read()

        mask = segment(frame, lower, upper)
        _,thresh = cv2.threshold(mask,127,255,0)

        hand_contour = get_my_hand(thresh, return_only_contour=True)
        hull = cv2.convexHull(hand_contour)
        final_image = np.zeros(frame.shape, np.uint8)
        cv2.drawContours(final_image, [hand_contour], 0, (0, 255, 0), 2)
        
        M = cv2.moments(hand_contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if prev_x is 0 and prev_y is 0: prev_x, prev_y = 0, 0
        
        delta_x, delta_y, slope, direction = prev_x-cx, prev_y-cy, 0, 'No movement'

        if delta_x**2+delta_y**2 > THRESHOLD**2:
            if delta_x is 0 and delta_y > 0: slope = 999 # inf
            elif delta_x is 0 and delta_y < 0: slope = -999 # -inf
            else: slope = float(delta_y/delta_x)

            if slope < 1.0 and slope >= -1.0 and delta_x > 0: direction = 'Right'
            elif slope < 1.0 and slope >= -1.0: direction = 'Left'
            elif (slope >= 1.0 or slope <=-1.0) and delta_y > 0.0: direction = 'Up'
            elif slope >= 1.0 or slope <=-1.0: direction = 'Down'

            THRESHOLD = 7
            prev_x, prev_y = cx, cy
        else:
            # Classification here
            direction = 'No movement'
            THRESHOLD = 20
        
        print('%-11s'%(direction), end='\r')

        cv2.imshow('Tracking hands', final_image)
        
        #print('Time per frame: '+str((time.time()-start_time)*1000)+'ms\r',end='')

        k = cv2.waitKey(100)
        if k == ord('q'):
            break
        elif k==ord('s'):
            contour_start=not contour_start
        elif k==ord('c'):
            cv2.imwrite('capture.jpg',frame)
    except Exception as e:
        print(e)
        break
cap.release()
cv2.destroyAllWindows()
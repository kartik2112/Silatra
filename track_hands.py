import cv2, numpy as np, time, math
from utils import segment, get_my_hand

#Open Camera object
cap = cv2.VideoCapture(0)
cap.set(3,640); cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS, 20)

contour_start=False
f = open('bounds.txt')
param = int(f.read().strip())
f.close()
lower = np.array([0,param,60],np.uint8)
upper = np.array([255,180,127],np.uint8)
while(1):
    start_time = time.time()
    ret, frame = cap.read()

    mask = segment(frame, lower, upper)
    _,thresh = cv2.threshold(mask,127,255,0)

    hand_contour = get_my_hand(thresh, return_only_contour=True)
    hull = cv2.convexHull(hand_contour)
    final_image = np.zeros(frame.shape, np.uint8)
    cv2.drawContours(final_image, [hand_contour], 0, (0, 255, 0), 2)
    #cv2.drawContours(final_image, [hull], 0, (0, 0, 255), 3)

    (x,y),(major_axis,minor_axis),angle = cv2.fitEllipse(hand_contour)
    cv2.ellipse(final_image,(int(x),int(y)),(int(major_axis/2),int(minor_axis/2)),angle,0.0,360.0,(0,255,0),2)
    ''' 
    # Counting fingers' code
    hull = cv2.convexHull(hand_contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(hand_contour, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(hand_contour[s][0])
                end = tuple(hand_contour[e][0])
                far = tuple(hand_contour[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(final_image, far, 8, [211, 84, 0], -1) '''
    cv2.imshow('Tracking hands', final_image)
    #print(str(cnt)+'\r',end='')
        
    print('Time per frame: '+str((time.time()-start_time)*1000)+'ms\r',end='')

    #close the output video by pressing 'ESC'
    k = cv2.waitKey(50) & 0xFF
    if k == ord('q'):
        break
    elif k==ord('s'):
        contour_start=not contour_start
    elif k==ord('c'):
        cv2.imwrite('capture.jpg',frame)
cap.release()
cv2.destroyAllWindows()
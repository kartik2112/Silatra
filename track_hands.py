import cv2, numpy as np, time, math

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

    blur = cv2.blur(frame,(3,3))
    ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)

    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(ycrcb,lower,upper)
    ret,thresh = cv2.threshold(mask2,127,255,0)

    #Find contours of the filtered frame
    _,contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   

    cv2.imshow('Skin segmentation using YCrCb mask',mask2)

    if contour_start:
        # Borrowed from - https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python
        
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            final_image = np.zeros(frame.shape, np.uint8)
            cv2.drawContours(final_image, [res], 0, (0, 255, 0), 2)
            #cv2.drawContours(final_image, [hull], 0, (0, 0, 255), 3)
            

            (x,y),(major_axis,minor_axis),angle = cv2.fitEllipse(res)
            cv2.ellipse(final_image,(int(x),int(y)),(int(major_axis/2),int(minor_axis/2)),angle,0.0,360.0,(0,255,0),2)
            ''' 
            # Counting fingers' code
            hull = cv2.convexHull(res, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(res, hull)
                if type(defects) != type(None):  # avoid crashing.   (BUG not found)

                    cnt = 0
                    for i in range(defects.shape[0]):  # calculate the angle
                        s, e, f, d = defects[i][0]
                        start = tuple(res[s][0])
                        end = tuple(res[e][0])
                        far = tuple(res[f][0])
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
''' 
# After skin segmentation
if not start_tracking:
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
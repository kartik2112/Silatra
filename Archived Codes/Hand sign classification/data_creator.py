import cv2, numpy as np, time, math

#Open Camera object
cap = cv2.VideoCapture(0)
cap.set(3,640); cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS, 20)

f, dump_file = open('bounds.txt'), open('khup chan.csv','a')
param = int(f.read().strip())
f.close()

contour_start=False
lower = np.array([0,137,60],np.uint8)
upper = np.array([255,180,127],np.uint8)
label, total_captured = 2, 0

while(1):
    start_time = time.time()
    ret, frame = cap.read()
    HEIGHT, WIDTH, _ = frame.shape

    blur = cv2.blur(frame,(3,3))
    ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)

    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(ycrcb,lower,upper)
    ret,thresh = cv2.threshold(mask2,127,255,0)

    #Find contours of the filtered frame
    _,contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   

    if not contour_start: cv2.imshow('Skin segmentation using YCrCb mask',mask2)
    else:
        try:
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
                cv2.drawContours(final_image, [hull], 0, (0, 0, 255), 3)
                
            # Create a ROI
            x1,y1,w1,h1 = cv2.boundingRect(hull)
            cv2.rectangle(final_image, (x1,y1), (x1+w1,y1+h1), (255,0,0), thickness=2)

            (x,y),(major_axis,minor_axis),angle = cv2.fitEllipse(hull)
            cv2.ellipse(final_image,(int(x),int(y)),(int(major_axis/2),int(minor_axis/2)),angle,0.0,360.0,(0,255,0),2)

            cv2.imshow('Tracking hands', final_image)
            
            center = (x-x1,y-y1)
            eccentricity = (1 - (major_axis/minor_axis) ** 2 ) ** 0.5
            scale = minor_axis*WIDTH/(major_axis*HEIGHT)
            act_r = ((center[0])**2+(center[1])**2)**0.5
            r = act_r/scale
            print('R: %4.3f, Scale: %2.3f, Eccentricity: %2.3f, (cx,cy): (%3d,%3d), Captured: %3d\r' % (act_r,scale,eccentricity,center[0],center[1],total_captured),end='')
        except:
            final_image = np.zeros(frame.shape, np.uint8)
            cv2.putText(final_image, 'Cannot find hand', (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), thickness=2)
            cv2.imshow('Tracking hands', final_image)
            continue

    k = cv2.waitKey(50) & 0xFF
    if k == ord('q'):
        break
    elif k==ord('s'):
        contour_start=not contour_start
        cv2.destroyAllWindows()
    elif k==ord('c'):
        if total_captured == 500: continue
        angle_90 = 180 - angle if angle > 90 else angle
        to_write_data = '%.3f,%d,%.3f,%d,%d,,%d,%.3f,%.3f,,%d\n' % (r,angle,eccentricity,center[0],center[1],angle_90,act_r,scale,label)
        dump_file.write(to_write_data)
        total_captured+=1

cap.release()
cv2.destroyAllWindows()
dump_file.close()
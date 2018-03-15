import numpy as np
import cv2
import imutils

prev_x, prev_y = 0, 0
THRESHOLD = 20

def trackDirection(hand_contour):
    global prev_x, prev_y, THRESHOLD
    M = cv2.moments(hand_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    if prev_x is 0 and prev_y is 0: prev_x, prev_y = 0, 0
    
    delta_x, delta_y, slope, direction = prev_x-cx, prev_y-cy, 0, 'None'

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
        return direction
    else:
        THRESHOLD = 20
        return 'None'
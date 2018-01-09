import cv2
import numpy
import keras

# print("Hello from python")
def multiply(a,b):
    print("Will compute", a, "times", b)
    c = 0
    for i in range(0, a):
        c = c + b
    return c

def displayImageParams(img):
    print("Image shape found from python:",img.shape)
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
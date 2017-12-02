'''
This to be done:

#Issue-1    Decide on Erosion & dilation

Need to figure out a way to speed up prediction.
'''

# Imports
from keras.models import model_from_json, Model
from keras import backend as k
from numpy import array,uint8
from keras.activations import relu
from math import floor
from PIL import Image
from os import system
import cv2, time, numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", 
	help = "If image is portrait, include this flag")
ap.add_argument("-i","--image", help='Use this flag followed by image file to do segmentation on an image')
args = vars(ap.parse_args())

print('\n--------------- Silatra skin detector ---------------')

# Read model architecture
model_data = ''
with open('model.json') as model_file: model_data = model_file.read()
model = model_from_json(model_data)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load saved weights
model.load_weights('weights.h5')
print('\nModel ready for testing. ',end='')

def predict_skin_pixels(img_file, return_flag=False):
    if img_file is not '': img_file = 'Test_Images/'+img_file

    # Start timer
    start = time.clock()

    # Load image & resize it to 640x480 pixels.
    if img_file is '': img_file = 'Test_Images/test_img.jpg'
    img, segmented_img, completed = cv2.imread(img_file), [], 0
    if not args.get('p'): img = cv2.resize(img, (320,240))                 # 240x320 resized image for faster prediction.
    else: img = cv2.resize(img, (240,320))
    original = img.copy()


    # Conversion to HSV & Normalization of image pixels
    ranges = [179.0,255.0,255.0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.tolist()
    for i in range(len(img)):                                   # Each row
        for j in range(len(img[i])):                            # Each pixel
            for k in range(3):                                  # Each channel (h/s/v)
                img[i][j][k] = img[i][j][k]*1.0/ranges[k]

    total_pixels = len(img)*len(img[0])
    print('Image size = '+str(len(img))+'x'+str(len(img[0]))+' = '+str(total_pixels)+' pixels\n')

    print('Processing image....\r',end='')
    for a_row in img:
        if a_row == [0,0,0]: continue
        output = model.predict(array(a_row))                    # Model needs a numpy array
        output = output.tolist()                                # Prediction is a numpy array. Convert to list for iteration
        pixel_vals = []
        for i in range(len(output)):
            if output[i][0]>output[i][1]:
                pixel_vals.append([1,1,1])
            else:
                pixel_vals.append([0,0,0])
        #completed += len(a_row)
        #if completed%10000 == 0: print('Completed: '+str(int(completed/10000))+"k/"+str(int(total_pixels/10000))+"k\r",end='')
        segmented_img.append(pixel_vals)
    print('Skin segmented from image.')

    # Bitwise and operation.
    ranges = [0,0,255]
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(3): img[i][j][k] = float(ranges[k]*segmented_img[i][j][k])

    ''' #Issue-1
    Unusre whether this code must be kept. 
    Some cases provides excellent results, but in some cases (particularly in cases of small hands) damages detected skin.

    # Erosion & dilation
    img = array(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    '''

    end = time.clock()
    print('Time required for segmentation: '+str(round(end-start,3))+'s')

    img = array(img)
    if not return_flag: cv2.imshow('Segmentation results',np.hstack([original, cv2.cvtColor(array(img, uint8), cv2.COLOR_HSV2BGR)]))
    else: return cv2.cvtColor(array(img, uint8), cv2.COLOR_HSV2BGR)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

if not args.get('image'):
    print('Keep image file empty for test image.')
    image_file = input('Image file: ')
    predict_skin_pixels(image_file)
    while True:
        continue_param = input('\n-----------------------------------------------------\n\nTest one more? (Y/N): ')
        print()
        if continue_param.lower() == 'y':
            image_file = input('Image file: ')
            predict_skin_pixels(image_file)
        else: break
else:
    print('\n')
    image_file = args.get('image')
    print('Using: '+image_file)
    cv2.imshow('Results',predict_skin_pixels(image_file, True))
    cv2.waitKey(10000)

'''
This code is kept if we want to test on individual RGB values. Remove this later.

# Sample prediction
r,g,b = 129,117,91
data_bgr = [b,g,r]
data_hsv = cv2.cvtColor(uint8([[data_bgr]]), cv2.COLOR_BGR2HSV).tolist()[0][0]
ranges = [255.0,100.0,100.0]
for i in range(len(data_to_test)): data_to_test[i] = data_to_test[i]*1.0/ranges[i]
data_to_test = [data_to_test]
output = model.predict(array(data_to_test)).tolist()
print('Output of model: '+str(output))
class_dt=""
if output[0][0]>output[0][1]:
    class_dt="Skin"
else:
    class_dt="Non-Skin"
print("Predicted class:"+class_dt)
'''
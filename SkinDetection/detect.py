'''
This to be done:

#Issue-1    Decide on Erosion & dilation
#Issue-2    Figure out a way to show images.

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

# Read model architecture
model_data = ''
with open('new_skin_model.json') as model_file: model_data = model_file.read()
model = model_from_json(model_data)
# model.summary() # Uncomment this to see overview of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load saved weights
model.load_weights('new_skin_model_weights.h5')
print('\nLoaded model\n')

# Start timer
start = time.clock()

# Load image & resize it to 640x480 pixels.
#img_file = '..\\training-images\\Digits\\5\\Right_Hand\\Normal\\10.png'
img_file = 'Test_Images/test2.jpg'
img, segmented_img, completed = cv2.imread(img_file), [], 0
img = cv2.resize(img, (320,240))                                # 240x320 resized image for faster prediction.

# Conversion to HSV & Normalization of image pixels
ranges = [255.0,100.0,100.0]
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = img.tolist()
for i in range(len(img)):                                   # Each row
    for j in range(len(img[i])):                            # Each pixel
        for k in range(3):                                  # Each channel (h/s/v)
            img[i][j][k] = img[i][j][k]*1.0/ranges[k]

total_pixels = len(img)*len(img[0])
print('Image size = '+str(len(img))+'x'+str(len(img[0]))+' = '+str(total_pixels)+' pixels\n')

print('Segmentation starts now.')
for a_row in img:
    output = model.predict(array(a_row))                    # Model needs a numpy array
    output = output.tolist()                                # Prediction is a numpy array. Convert to list for iteration
    pixel_vals = []
    for i in range(len(output)):
        if output[i][0]>output[i][1]:
            pixel_vals.append([1,1,1])
        else:
            pixel_vals.append([0,0,0])
    completed += len(a_row)
    print('Completed: '+str(completed)+"/"+str(total_pixels)+"\r",end='')
    segmented_img.append(pixel_vals)

# Bitwise and operation. Do not use inbuilt cv2 function as it won't work with 2 channels.
for i in range(len(img)):
    for j in range(len(img[i])):
        for k in range(3): img[i][j][k] *= int(ranges[k]*segmented_img[i][j][k])

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

''' #Issue-2
Need to find out a way to show the image. 
The image window hangs when this function comes up with a new window.

#cv2.imshow('Post segmentation',array(img))
'''

# These lines are for testing purposes only. Remove them later.
cv2.imwrite('segmented.jpg',array(img))
system('start segmented.jpg')
system('start '+img_file)

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
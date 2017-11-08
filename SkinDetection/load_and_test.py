from keras.models import model_from_json, Model
from keras import backend as k
from numpy import array,uint8
from keras.activations import relu
from math import floor
from PIL import Image
import cv2, time, numpy as np

# Start timer
start = time.clock()

# Read model architecture
model_data = ''
with open('skin.json') as model_file: model_data = model_file.read()
model = model_from_json(model_data)
# model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load saved weights
model.load_weights('skin.h5')

print('\nLoaded model\n')


img, segmented_img, completed = cv2.imread('Test_Images/hand.jpg'), [], 0
img = cv2.resize(img, (320,240))
total_pixels = len(img)*len(img[0])

# Conversion to HSV & Normalization of image pixels
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = img.tolist()

for i in range(len(img)):                                   # Each row
    for j in range(len(img[i])):                            # Each pixel
        img[i][j] = img[i][j][0:2]
        for k in range(2):                                  # Each channel (r/g/b)
            img[i][j][k] = img[i][j][k]*1.0/255.0

print('Image size = '+str(len(img))+'x'+str(len(img[0]))+' = '+str(total_pixels)+' pixels\n')

# Prediction starts here
print('Segmentation starts now.')
for a_row in img:
    output = model.predict(array(a_row))                    # Model needs a numpy array
    output = output.tolist()
    pixel_vals = []
    for i in range(len(output)):
        if output[i][0]>output[i][1]:
            pixel_vals.append([1,1])
        else:
            pixel_vals.append([0,0])
    completed += len(a_row)
    print('Completed: '+str(completed)+"/"+str(total_pixels)+"\r",end='')
    segmented_img.append(pixel_vals)

for i in range(len(img)):
    for j in range(len(img[i])):
        for k in range(2): img[i][j][k] *= int(255*segmented_img[i][j][k])
        img[i][j].append(10)

img = array(img)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
img = cv2.dilate(img, kernel, iterations=4)
img = cv2.erode(img, kernel, iterations=3)

end = time.clock()
print('Time required for segmentation: '+str(round(end-start,3))+'s')
#cv2.imshow('Post segmentation',array(img))
cv2.imwrite('segmented.jpg',array(img))



''' 
# Sample prediction
r,g,b = 129,117,91
data_bgr = [b,g,r]
data_hsv = cv2.cvtColor(uint8([[data_bgr]]), cv2.COLOR_BGR2HSV).tolist()[0][0]
data_to_test = data_hsv[1:]
for i in range(len(data_to_test)): data_to_test[i] = data_to_test[i]*1.0/255.0
data_to_test = [data_to_test]
output = model.predict(array(data_to_test)).tolist()
print('Output of model: '+str(output))
class_dt=""
if output[0][0]>output[0][1]:
    class_dt="Skin"
else:
    class_dt="Non-Skin"
print("Predicted class:"+class_dt) '''

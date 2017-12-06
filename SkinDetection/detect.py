'''
This to be done:

#Issue-1    Decide on Erosion & dilation

Need to figure out a way to speed up prediction.
'''

# Imports
from keras.models import model_from_json, Model
from keras import backend as k
from numpy import array,uint8
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

#Set Segmentation threshold
SEGMENTATION_THRESHOLD = 0.5

with open('probabilities.txt') as p:
    probability_of_being_skin_pixel, count = 0, 0
    while True:
        try:
            line = p.read()
            if line is '': break
            probability_of_being_skin_pixel += float(line.split('\n')[0])
            count += 1
        except: break

probability_of_being_skin_pixel /= count

def predict_skin_pixels(img_file, return_flag=False):
    if img_file is not '': img_file = 'Test_Images/'+img_file

    # Start timer
    start = time.clock()

    # Load image
    if img_file is '': img_file = 'Test_Images/test_img.jpg'
    img, segmented_img, completed = cv2.imread(img_file), [], 0

    # Decide aspect ratio and resize the image.
    if float(len(img)/len(img[0])) == float(16/9): img = cv2.resize(img, (240,320))
    elif float(len(img)/len(img[0])) == float(9/16): img = cv2.resize(img, (320,240))
    elif float(len(img)/len(img[0])) == 1: img = cv2.resize(img, (320,240))
    else: img = cv2.resize(img, (250,250))
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

    # Classification of pixels to skin and non-skin
    total_skin_pixels = 0
    print('Processing image....\r',end='')
    for a_row in img:
        if a_row == [0,0,0]: continue
        output = model.predict(array(a_row))                    # Model needs a numpy array
        output = output.tolist()                                # Prediction is a numpy array. Convert to list for iteration
        pixel_vals = []
        for i in range(len(output)):
            if output[i][0]*probability_of_being_skin_pixel >= output[i][1]*(1.0-probability_of_being_skin_pixel):
            #if output[i][0] > output[i][1]:
                pixel_vals.append([1,1,1])
                total_skin_pixels += 1
            else:
                pixel_vals.append([0,0,0])
        segmented_img.append(pixel_vals)
    print('Skin segmented from image.')

    # Bitwise AND operation to binarise the image.
    ranges = [0,0,255]
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(3): img[i][j][k] = float(ranges[k]*segmented_img[i][j][k])
    
    end = time.clock()
    print('Time required for segmentation: '+str(round(end-start,3))+'s')

    #print('Probability of a pixel being skin: '+str(round(float(total_skin_pixels)/float(total_pixels),3)))
    #with open('probabilities.txt','a') as f: f.write(str(round(float(total_skin_pixels)/float(total_pixels),3))+'\n')

    img = array(img)
    if not return_flag: cv2.imshow('Segmentation results',np.hstack([original, cv2.cvtColor(array(img, uint8), cv2.COLOR_HSV2BGR)]))
    else: return cv2.cvtColor(array(img, uint8), cv2.COLOR_HSV2BGR)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

if not args.get('image'):
    print('Keep image file empty for test image.\n')
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
    cv2.waitKey(100000)

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
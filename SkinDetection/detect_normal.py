from keras.models import model_from_json
import numpy as np
import cv2, time
import argparse
from math import ceil

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", help='Use this flag followed by image file to do segmentation on an image from any folder (use absolute path)')
ap.add_argument("-ts","--test_image", help='Use this flag followed by image file to do segmentation on an image from Test_Images folder')
ap.add_argument("-tr","--train_image", help='Use this flag followed by image file to do segmentation on an image from training-images folder')
args = vars(ap.parse_args())

model_data = ''
with open('model1.json') as model_file: model_data = model_file.read()
model = model_from_json(model_data)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load saved weights
model.load_weights('weights1.h5')
print('\nModel ready for testing. ',end='')
start = time.clock()
img=''
if args.get('image'):img=input()
elif args.get('test_image'):img='Test_Images/'+input()
elif args.get('train_image'):img='../training-images/Digits/'+input()
# if float(len(img)/len(img[0])) == float(16/9): img = cv2.resize(img, (180,320))
# elif float(len(img)/len(img[0])) == float(9/16): img = cv2.resize(img, (320,180))
# elif float(len(img)/len(img[0])) == float(4/3): img = cv2.resize(img, (320,240))
# elif float(len(img)/len(img[0])) == float(3/4): img = cv2.resize(img, (240,320))
# elif float(len(img)/len(img[0])) == 1: img = cv2.resize(img, (300,300))
# else: img = cv2.resize(img, (250,250))
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = img.tolist()
seg_img=[]
for row in img:
    output=model.predict(row)
    pixel_vals=[]
    for prediction in output:
        if prediction[0]>=prediction[1]:
            pixel_vals.append([255.0,255.0,255.0])
        else:
            pixel_vals.append([0.0,0.0,0.0])
    seg_img.append(pixel_vals)

seg_img=np.array(seg_img)
end = time.clock()
print('Time required for segmentation: '+str(round(end-start,3))+'s')
cv2.imshow('Segmentated image',seg_img)
cv2.imwrite("../Results and ROC/DNN model-BW.jpg",seg_img)
cv2.waitKey(100000)
cv2.destroyAllWindows()

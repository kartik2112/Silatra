from keras.models import model_from_json, Model
from keras import backend as k
from numpy import array
from keras.activations import relu
from math import floor
import cv2, time

# Start timer
start = time.clock()

# Read model architecture
model_data = ''
with open('skin.json') as model_file: model_data = model_file.read()
model = model_from_json(model_data)
model.summary()
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

# Load saved weights
model.load_weights('skin.h5')

print '\nLoaded model\n'

img, segmented_img, completed = cv2.imread('1.png').tolist(), [], 0
total_pixels = len(img)*len(img[0])

# Normalization of image pixels
for i in range(len(img)):                                   # Each row
    for j in range(len(img[i])):                            # Each pixel
        for k in range(3):                                  # Each channel (r/g/b)
            img[i][j][k] = img[i][j][k]*1.0/255.0

print 'Image size = '+str(len(img))+'x'+str(len(img[0]))+' = '+str(total_pixels)+' pixels\n'

# Prediction starts here
for a_row in img:
    output = model.predict(array(a_row))                    # Model needs a numpy array
    
    completed += len(a_row)
    print 'Completed: '+str(completed)+"/"+str(total_pixels)+"\r",
    segmented_img.append(output)

end = time.clock()
print 'Time required for segmentation: '+str(end-start)
''' cv2.imshow('Segmented image',array(segmented_img))
raw_input()
cv2.destroyAllWindows() '''
#print segmented_img



'''
# Sample prediction

r,g,b = 203.0,213.0,253.0
r,g,b = r/255.0, g/255.0, b/255.0
data = [[r,g,b]]
data = array(data)
output = model.predict(data)
print 'Input: (',r,',',g,',',b,') output:',floor(output[0][0])
 '''

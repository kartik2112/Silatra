from keras.models import model_from_json
from keras import backend as k
import cv2

''' # Read model architecture
model_data = ''
with open('skin.json') as model_file: model_data = model_file.read()
model = model_from_json(model_data)

# Load saved weights
model.load_weights('skin.h5')

print 'Loaded model'
model.summary() '''

''' 
img = cv2.imread('1.png')
segmented_img = []
for a_row in img:
    for pixel in a_row:
        # Dealing with each individual pixel.
'''
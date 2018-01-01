# Imports
from keras.models import model_from_json
from numpy import array,uint8
import cv2

# Read model architecture
model_data = ''
with open('SkinDetection/model.json') as model_file: model_data = model_file.read()
model = model_from_json(model_data)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load saved weights
model.load_weights('SkinDetection/weights.h5')

def predict_skin_pixels(img_file='./SkinDetection/Test_Images/test_img.jpg'):
    img, segmented_img, completed = cv2.imread(img_file), [], 0

    # Decide aspect ratio and resize the image.
    if float(len(img)/len(img[0])) == float(16/9): img = cv2.resize(img, (180,320))
    elif float(len(img)/len(img[0])) == float(9/16): img = cv2.resize(img, (320,180))
    elif float(len(img)/len(img[0])) == float(4/3): img = cv2.resize(img, (320,240))
    elif float(len(img)/len(img[0])) == float(3/4): img = cv2.resize(img, (240,320))
    elif float(len(img)/len(img[0])) == 1: img = cv2.resize(img, (300,300))
    else: img = cv2.resize(img, (250,250))

    # Conversion to HSV & Normalization of image pixels
    ranges = [179.0,255.0,255.0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.tolist()
    for i in range(len(img)):                                   # Each row
        for j in range(len(img[i])):                            # Each pixel
            for k in range(3):                                  # Each channel (h/s/v)
                img[i][j][k] = img[i][j][k]*1.0/ranges[k]

    completed, n, total_pixels = 0, len(img[0]), len(img)*len(img[0])
    upper_row_predictions, curr_row_predictions, lower_row_predictions = '', model.predict(img[0]), model.predict(img[1])
    k1, K = 1.0, 1
    for i in range(len(img)):
        for j in range(len(curr_row_predictions)):
            l_skin, count = 0.0, 0
            if i is not 0:
                count += 1
                if j > 0:
                    l_skin = upper_row_predictions[j-1][0]
                    count += 1
                l_skin += upper_row_predictions[j][0]
                if j < n-1:
                    l_skin += upper_row_predictions[j+1][0]
                    count += 1
            if j > 0:
                l_skin += curr_row_predictions[j-1][0]
                count += 1
            if j < n-1:
                l_skin += curr_row_predictions[j+1][0]
                count += 1
            if i is not len(img)-1:
                count += 1
                if j > 0:
                    l_skin += lower_row_predictions[j-1][0]
                    count += 1
                l_skin += lower_row_predictions[j][0]
                if j < n-1:
                    l_skin += lower_row_predictions[j+1][0]
                    count += 1
            alpha = l_skin
            l_skin = l_skin*k1/(1.0*count)
            if curr_row_predictions[j][0]*l_skin >= 0.5:
                for k in range(3): img[i][j][k] *= float(ranges[k])
                k1 = count*1.0*K/alpha
            else:
                img[i][j] = [0.0,0.0,0.0]
                k1 = 1
            completed += 1
            if completed%10000 == 0: print('Skin segmentation in process: '+str(completed/1000)+'K / '+str(total_pixels/1000)+'K\r',end='')
        upper_row_predictions = curr_row_predictions
        curr_row_predictions = lower_row_predictions
        if i < len(img)-2: lower_row_predictions = model.predict(img[i+2])
    print('                                                                                                                                \r',end='')
    return cv2.cvtColor(array(img, uint8), cv2.COLOR_HSV2BGR)
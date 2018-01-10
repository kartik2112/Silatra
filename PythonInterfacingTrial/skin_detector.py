# Imports
from keras.models import model_from_json
from numpy import array,uint8
import cv2
from math import ceil

# Read model architecture
model_data = ''
with open('../SkinDetection/deep_model.json') as model_file: model_data = model_file.read()
model = model_from_json(model_data)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load saved weights
model.load_weights('../SkinDetection/deep_weights.h5')

def segment(img, return_mask=False):

    # Decide aspect ratio and resize the image.
    if float(len(img)/len(img[0])) == float(16/9): img = cv2.resize(img, (180,320))
    elif float(len(img)/len(img[0])) == float(9/16): img = cv2.resize(img, (320,180))
    elif float(len(img)/len(img[0])) == float(4/3): img = cv2.resize(img, (160,120))
    elif float(len(img)/len(img[0])) == float(3/4): img = cv2.resize(img, (160,120))
    elif float(len(img)/len(img[0])) == 1: img = cv2.resize(img, (300,300))
    else: img = cv2.resize(img, (250,250))
    original = img.copy()
    h,w,_ = img.shape
    img = cv2.resize(img, (ceil(w/2), ceil(h/2)))

    # Conversion to HSV & Normalization of image pixels
    ranges = [179.0,255.0,255.0]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.tolist()
    for i in range(len(img)):                                   # Each row
        for j in range(len(img[i])):                            # Each pixel
            for k in range(3):                                  # Each channel (h/s/v)
                img[i][j][k] = img[i][j][k]*1.0/ranges[k]

    total_pixels = len(img)*len(img[0])

    completed, n = 0, len(img[0])
    segmented_img = []
    upper_row_predictions, curr_row_predictions, lower_row_predictions = '', model.predict(img[0]), model.predict(img[1])
    k1, K = 1.0, 1
    mask = []
    for i in range(len(img)):
        mask_row = []
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
                mask_row.append([0.001,0.001,255.0])
                k1 = count*1.0*K/alpha
            else:
                mask_row.append([0.0,0.0,0.0])
                k1 = 1
            completed += 1
        upper_row_predictions = curr_row_predictions
        curr_row_predictions = lower_row_predictions
        mask.append(mask_row)
        if i < len(img)-2: lower_row_predictions = model.predict(img[i+2])
    mask = cv2.resize(array(mask), (w, h))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask, segmented_img = mask.tolist(), cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    for i in range(len(segmented_img)):
        for j in range(len(segmented_img[i])):
            for k in range(3):
                if mask[i][j][k] == 0.0: segmented_img[i][j][k] = 0.0
    if not return_mask: return cv2.cvtColor(array(segmented_img, uint8), cv2.COLOR_HSV2BGR)
    else: return cv2.cvtColor(array(mask, uint8), cv2.COLOR_HSV2BGR)
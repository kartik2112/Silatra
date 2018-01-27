'''
To be run after:
create_data.py & then build_model.py

Segments skin from an image. Input image is taken from Test_Images folder as of now.
'''
''' Determination of pixels - skin and non-skin

    How probability of a pixel being skin or non-skin is calculated:
    The probability that a pixel is skin pixel depends on following factors:
    1. Whether colour of pixel is skin colour
    2. Whether neighbouring colours are skin (Belonging in a skin area)

    Thus, we can decide probability of a pixel being skin by using following equation:
    P(pixel: skin) = P(pixel: skin_colour) x L(pixel: skin)

    where L(pixel: skin) represents the likelihood of pixel being skin.
    More the chance of neighbouring pixels being of skin colour, more the chance of current pixel being skin.abs
    Thus, we calculate L(pixel: skin) by taking average of the probabilities of being skin colour of the 8 neighbouring pixels.
    L(pixel: skin) = Avg(P(pixel-i: skin_colour)) where pixel-i is one of the 8 neighbouring pixel.

'''

# Imports
from keras.models import model_from_json
from numpy import array,uint8,hstack,reshape
import cv2, time
import argparse
from math import ceil

# Add support for using flags such as -i and --image for direct image input
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", help='Use this flag followed by image file to do segmentation on an image from any folder (use absolute path)')
ap.add_argument("-ts","--test_image", help='Use this flag followed by image file to do segmentation on an image from Test_Images folder')
ap.add_argument("-tr","--train_image", help='Use this flag followed by image file to do segmentation on an image from training-images folder')
ap.add_argument("-nr","--no_resize",help="Use this flag if you don't want the resize feature to speed up processing.")
args = vars(ap.parse_args())

print('\n--------------- Silatra skin detector ---------------')

# Read model architecture
model_data = ''
with open('model1.json') as model_file: model_data = model_file.read()
model = model_from_json(model_data)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load saved weights
model.load_weights('weights1.h5')
print('\nModel ready for testing. ',end='')

def predict_skin_pixels(img_file, return_flag=False):
    if img_file is not '' and not args.get('image') and not args.get('test_image'): img_file = '../training-images/Digits/'+img_file
    elif args.get('test_image'): img_file = 'Test_Images/'+img_file

    # Start timer
    start = time.clock()

    # Load image
    if img_file is '': img_file = 'Test_Images/test_img.jpg'
    img, segmented_img, completed = cv2.imread(img_file), [], 0

    # Decide aspect ratio and resize the image.
    # if float(len(img)/len(img[0])) == float(16/9): img = cv2.resize(img, (180,320))
    # elif float(len(img)/len(img[0])) == float(9/16): img = cv2.resize(img, (320,180))
    # elif float(len(img)/len(img[0])) == float(4/3): img = cv2.resize(img, (320,240))
    # elif float(len(img)/len(img[0])) == float(3/4): img = cv2.resize(img, (240,320))
    # elif float(len(img)/len(img[0])) == 1: img = cv2.resize(img, (300,300))
    # else: img = cv2.resize(img, (250,250))
    original = img.copy()
    h,w,_ = img.shape
    if not args.get('no_resize'): img = cv2.resize(img, (ceil(w/3), ceil(h/3)))

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

    completed, n = 0, len(img[0])
    t1 = time.time()
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
            if completed%10000 == 0: print(str(completed/1000)+'K / '+str(total_pixels/1000)+'K\r',end='')
        upper_row_predictions = curr_row_predictions
        curr_row_predictions = lower_row_predictions
        mask.append(mask_row)
        if i < len(img)-2: lower_row_predictions = model.predict(img[i+2])
    print(str(total_pixels/1000)+'K / '+str(total_pixels/1000)+'K\r',end='')
    print('Skin segmented from image.')
    t2 = time.time()
    print('Time required for actual segmentation -> '+str(t2-t1)+' seconds')

    mask = array(mask)
    if not args.get('no_resize'): mask = cv2.resize(mask, (w, h))
    ''' kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kernel) '''
    cv2.imshow('Mask',cv2.cvtColor(array(mask, uint8), cv2.COLOR_HSV2BGR))

    mask, segmented_img = mask.tolist(), cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    for i in range(len(segmented_img)):
        for j in range(len(segmented_img[i])):
            for k in range(3):
                if mask[i][j][k] == 0.0: segmented_img[i][j][k] = 0.0
    # Stop timer and measure the time for segmentation
    end = time.clock()
    print('Time required for segmentation: '+str(round(end-start,3))+'s')

    # Show results
    if not return_flag: cv2.imshow('Segmentation results',hstack([original, cv2.cvtColor(array(segmented_img, uint8), cv2.COLOR_HSV2BGR)]))
    else: return cv2.cvtColor(array(img, uint8), cv2.COLOR_HSV2BGR)
    cv2.waitKey(100000)
    cv2.destroyAllWindows()

# If no flags specified, execution starts here
if not args.get('image') and not args.get('train_image') and not args.get('test_image'):
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
# When -i or --image flag is used, execution starts here.
else:
    print('\n')
    image_file = args.get('image')
    if not args.get('image') and not args.get('test_image'): image_file = args.get('train_image')
    elif not args.get('image'): image_file = args.get('test_image')
    print('Using: '+image_file)
    predict_skin_pixels(image_file)
    cv2.waitKey(100000)
# good hand.jpg: 11.9843 seconds

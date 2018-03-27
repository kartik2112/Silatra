import cv2, numpy as np, time
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from utils import extract_features, get_my_hand, segment
import silatra

lower = np.array([0,145,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

start = time.time()
dump_file = open('gesture_signs_selected.csv','a')
grid = (20,20)
# for i in range(grid[0]*grid[1]): dump_file.write('f'+str(i)+',')
# dump_file.write('label\n')

total_images_parsed = 0
DATA_LOCS = ['../training-images/Gesture_Signs/']
for loc in range(len(DATA_LOCS)):
    DATA_LOC = DATA_LOCS[loc]
    # for label in ['Apple_Finger','Cup_Closed','Cup_Open','Sorry_Fist','Sun_Up','That_is_Good_Circle','That_is_Good_Point','ThumbsUp']:
    for label in ['Cup_Closed','Cup_Open','Sun_Up','ThumbsUp']:
        for i in range(500,601):
            try:
                print(' '*160+'\rProcessing image: %3d, Label = %s, From Location: %s' % (i,label,DATA_LOC),end='\r')
                
                image = cv2.imread(DATA_LOC+str(label)+"/"+str(i)+'.png')

                if image.shape[0] == 0: continue
                
                mask,_,_ = silatra.segment(image)
                _,thresh = cv2.threshold(mask,127,255,0)

                hand = get_my_hand(thresh)
                features = extract_features(hand, grid)
                
                to_write_data = ''
                for feature in features: to_write_data += str(feature) + ','
                to_write_data += label + '\n'
                
                dump_file.write(to_write_data)
                total_images_parsed += 1
            except Exception as e:
                print(e)
                continue
total = (time.time() - start)
# print(' '*160+'\rTotal time required = %3.3fs' % (total))
print('Total images parsed: %d'%(total_images_parsed))
# winsound.Beep(1000, 100)
# winsound.Beep(1200, 100)
# winsound.Beep(1500, 100)
# winsound.Beep(1700, 100)
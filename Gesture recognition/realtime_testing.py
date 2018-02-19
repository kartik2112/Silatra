import cv2, numpy as np, time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from utils import extract_features, get_my_hand, segment

data = pd.read_csv('gesture_data.csv')

X = data[['f'+str(i) for i in range(400)]].values
Y = data['label'].values

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, Y)

cap = cv2.VideoCapture(0)
cap.set(3,640); cap.set(4,480)
cap.set(cv2.CAP_PROP_FPS, 20)

contour_start=False
lower = np.array([0,147,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

grid = (20,20)

print('gaf0 gaf1')
while(1):
    start_time = time.time()
    _, frame = cap.read()
    
    x,y,w,h = 100,100,300,300
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    roi = frame[y:y+h,x:x+w]

    mask = segment(roi, lower, upper)

    if not contour_start:
        cv2.imshow('Skin segmentation using YCrCb mask',mask)
        cv2.imshow('Original',frame)
    else:
        try:
            _,thresh = cv2.threshold(mask,127,255,0)
            hand = get_my_hand(thresh)
            
            cv2.imshow('Your hand',hand)
            cv2.imshow('Original',frame)

            features = extract_features(hand, grid)
            predictions = classifier.predict_proba([features]).tolist()[0]
            for prob in predictions: print('%.2f'%(prob),end=' ')
            print('',end='\r')
        except Exception as e:
            print(e,end='\r')
            final_image = np.zeros(frame.shape, np.uint8)
            cv2.putText(final_image, 'Cannot find hand', (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), thickness=2)
            cv2.imshow('Original', final_image)
            #continue

    k = cv2.waitKey(50)
    if k == ord('q'):
        break
    elif k==ord('s'):
        contour_start=not contour_start
        cv2.destroyAllWindows()
        
cap.release()
cv2.destroyAllWindows()
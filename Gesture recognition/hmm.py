import cv2, numpy as np
import hidden_markov as hm
from utils import segment, get_my_hand, extract_features
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('gesture_data.csv')

X = data[['f'+str(i) for i in range(400)]].values
Y = data['label'].values

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X,Y)

lower = np.array([0,147,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

cap = cv2.VideoCapture('good afternoon.avi')

obs = []
frame_no, grid = 1, (20,20)
print('Frame: [gaf0, gaf1]')
while(1):
    try:
        _,frame = cap.read()
        mask = segment(frame, lower, upper)
        _,thresh = cv2.threshold(mask,127,255,0)
        hand = get_my_hand(thresh)
        features = extract_features(hand,grid)
        pred = classifier.predict([features])
        print('%5d'%(frame_no),end=': ')
        print(pred)
        pred = pred.tolist()
        obs.append(pred)
        frame_no+=1
    except:
        break
cap.release()

print(obs)
states = ('gaf0', 'gaf1')
observations = ('0','1')
start_prob = np.matrix('0.5 0.5')
transition_prob = np.matrix('1.0 0.0 ; 0.0 1.0')
emission_prob = np.matrix('0.7 0.3 ; 1.0 0 ')

good_afternoon = hm.hmm(states, observations, start_prob, transition_prob, emission_prob)

observed = [('0','0','0','0','1','1')]
observed.extend(('0','0','1'))
e,t,s = good_afternoon.train_hmm(obs, 30, [10, 20])
print(e)
print(t)
print(s)
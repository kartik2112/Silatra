import numpy as np
import timeit
from hmmlearn import hmm
from sklearn.externals import joblib
import sys

sys.path.insert(0, "Gesture_Modules")
import hmmGestureClassify

ctr = 0
corr = 0

lastOne = 'Good Morning'

with open('../HMMTrainer/gestures.csv') as fileR:
    while True:
        line = fileR.readline()
        if line == '': break
        line = line.strip().split(',')
        class1 = line[0]
        if class1 != lastOne:
            print("\nAccuracy yet:",corr*100/ctr,"\n")
            lastOne = class1
            ctr = 0
            corr = 0
        print("\r"+class1,end='')
        line[1:] = map(int,line[1:])
        data = line[1:]
        trainTemp = np.reshape(np.array(data),(-1,1))
        ctr += 1
        class2 = hmmGestureClassify.hmmGestureClassify(trainTemp)[0]
        if class1 == class2:
            corr += 1
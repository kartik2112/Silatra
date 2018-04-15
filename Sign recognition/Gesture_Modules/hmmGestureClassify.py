import hmmlearn
import os
from sklearn.externals import joblib
import nose
from nose import tools
import numpy as np

dir = "./Models/GestureHMMs"

Models = []
ModelNames = []

kfMapper = {'Up':0,'Right':1,'Left':2,'Down':3,'ThumbsUp':4, 'Sun_Up':5, 'Cup_Open':6, 'Cup_Closed':7}

for model in os.listdir(dir):
    Models += [joblib.load(dir+"/"+model)]
    ModelNames += [model.split(".")[0]]

def classifyGestureByHMM(sequence):
    '''
    Here, sequence is a list of tuples of the form: <Sign, Direction>
    '''
    testInputSeq = []
    for elem in sequence:
        if elem[0] == 'None':
            testInputSeq += [kfMapper[elem[1]]]
        else:
            testInputSeq += [kfMapper[elem[0]]]
    maxScore = float('-inf')
    print(testInputSeq)
    testInputSeq = np.reshape(np.array(testInputSeq),(-1,1))
    # print(testInputSeq)
    recognizedGesture = "--"
    for i in range(len(Models)):
        scoreTemp = Models[i].score(testInputSeq)
        print(ModelNames[i],":",scoreTemp)
        if scoreTemp > maxScore:
            maxScore = scoreTemp
            recognizedGesture = ModelNames[i]
    # print((recognizedGesture,maxScore))
    return (recognizedGesture,maxScore)

def test_stable_series_of_Good_Afternoon():
    IP_Ts = [("ThumbsUp","None"),("ThumbsUp","None"),("ThumbsUp","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),("ThumbsUp","None"),
            ("ThumbsUp","None"),
            ("None","Up"),("None","Up"),("None","Up"),("None","Up"),
            ("Sun_Up","None"),("Sun_Up","None"),("Sun_Up","None"),
            ("Sun_Up","None"),("Sun_Up","None"),("Sun_Up","None"),
            ("Sun_Up","None")]

    Expected_TS = [("ThumbsUp","None"), ("None","Up") ,("Sun_Up","None")]
    Expected_Gesture = 'Good Afternoon'
    tools.eq_(classifyGestureByHMM(IP_Ts)[0],Expected_Gesture)

def test_noisy1_series_of_Good_Afternoon():
    IP_Ts = [("ThumbsUp","None"),("Cup_Closed","None"),("ThumbsUp","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),("ThumbsUp","None"),
            ("ThumbsUp","None"),
            ("None","Up"),("None","Up"),("None","Up"),("None","Up"),
            ("Sun_Up","None"),("Sun_Up","None"),("Sun_Up","None"),
            ("Sun_Up","None"),("Sun_Up","None"),("Cup_Open","None"),
            ("Sun_Up","None")]

    Expected_TS = [("ThumbsUp","None"), ("None","Up") ,("Sun_Up","None")]
    Expected_Gesture = 'Good Afternoon'
    tools.eq_(classifyGestureByHMM(IP_Ts)[0],Expected_Gesture)

def test_noisy2_series_of_Good_Afternoon():
    IP_Ts = [("ThumbsUp","None"),("None","Up"),("Cup_Closed","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),("ThumbsUp","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),
            ("None","Up"),("None","Up"),("None","Up"),("None","Up"),
            ("Sun_Up","None"),("Cup_Closed","None"),("Cup_Open","None"),
            ("Sun_Up","None"),("Sun_Up","None"),("Sun_Up","None"),
            ("Sun_Up","None")]

    Expected_TS = [("ThumbsUp","None"), ("None","Up") ,("Sun_Up","None")]
    Expected_Gesture = 'Good Afternoon'
    tools.eq_(classifyGestureByHMM(IP_Ts)[0],Expected_Gesture)

def test_noisy3_series_of_Good_Afternoon():
    IP_Ts = [("ThumbsUp","None"),("None","Up"),("Cup_Closed","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),("ThumbsUp","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),
            ("None","Up"),("ThumbsUp","None"),("None","Left"),("None","Up"),("None","Up"),
            ("Sun_Up","None"),("Sun_Up","None"),("Sun_Up","None"),
            ("Sun_Up","None"),("Sun_Up","None"),("Cup_Open","None"),
            ("Cup_Closed","None")]

    Expected_TS = [("ThumbsUp","None"), ("None","Up") ,("Sun_Up","None")]
    Expected_Gesture = 'Good Afternoon'
    tools.eq_(classifyGestureByHMM(IP_Ts)[0],Expected_Gesture)

def test_noisy4_series_of_Good_Afternoon():
    IP_Ts = [("ThumbsUp","None"),("None","Up"),("Cup_Closed","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),("ThumbsUp","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),
            ("None","Up"),("ThumbsUp","None"),
            ("None","Up"),("None","Up"),("None","Up"),
            ("Sun_Up","None"),("Sun_Up","None"),("Cup_Open","None"),
            ("Cup_Open","None"),("Sun_Up","None"),("Sun_Up","None"),
            ("Sun_Up","None")]

    Expected_TS = [("ThumbsUp","None"), ("None","Up") ,("Sun_Up","None")]
    Expected_Gesture = 'Good Afternoon'
    tools.eq_(classifyGestureByHMM(IP_Ts)[0],Expected_Gesture)

def test_stable_series_of_Good_Morning():
    IP_Ts = [("ThumbsUp","None"),("ThumbsUp","None"),("ThumbsUp","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),("ThumbsUp","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),
            ("None","Down"),("None","Down"),("None","Down"),("None","Down"),
            ("Cup_Closed","None"),("Cup_Closed","None"),("Cup_Closed","None"),
            ("Cup_Closed","None"),("Cup_Closed","None"),("Cup_Closed","None"),
            ("Cup_Closed","None"),
            ("None","Up"),("None","Up"),("None","Up"),("None","Up"),("None","Up"),("None","Up"),
            ("Cup_Open","None"),("Cup_Closed","None"),("Cup_Open","None"),
            ("Cup_Open","None"),("Cup_Open","None"),("Cup_Open","None"),
            ("Cup_Open","None")]

    Expected_TS = [("ThumbsUp","None"), ("None","Down"), ("Cup_Closed","None"), ("None","Up"), ("Cup_Open","None")]
    Expected_Gesture = 'Good Morning'
    tools.eq_(classifyGestureByHMM(IP_Ts)[0],Expected_Gesture)

def test_noisy1_series_of_Good_Morning():
    IP_Ts = [("ThumbsUp","None"),("None","Up"),("Cup_Closed","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),("ThumbsUp","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),
            ("None","Down"),("ThumbsUp","None"),
            ("None","Down"),("None","Down"),("None","Down"),("None","Down"),
            ("Cup_Closed","None"),("ThumbsUp","None"),("Cup_Closed","None"),
            ("Cup_Closed","None"),("ThumbsUp","None"),("Cup_Closed","None"),
            ("Cup_Closed","None"),
            ("None","Up"),("ThumbsUp","None"),
            ("None","Up"),("None","Up"),("None","Up"),("None","Up"),("None","Up"),
            ("Cup_Open","None"),("Cup_Closed","None"),("Cup_Open","None"),
            ("Cup_Open","None"),("Cup_Open","None"),("Cup_Open","None"),
            ("Cup_Closed","None")]

    Expected_TS = [("ThumbsUp","None"), ("None","Down"), ("Cup_Closed","None"), ("None","Up"), ("Cup_Open","None")]
    Expected_Gesture = 'Good Morning'
    tools.eq_(classifyGestureByHMM(IP_Ts)[0],Expected_Gesture)

def test_stable_series_of_Good_Night():
    IP_Ts = [("ThumbsUp","None"),("ThumbsUp","None"),("ThumbsUp","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),("ThumbsUp","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),
            ("None","Up"),("None","Up"),("None","Up"),("None","Up"),("None","Up"),("None","Up"),
            ("Cup_Open","None"),("Cup_Closed","None"),("Cup_Open","None"),
            ("Cup_Open","None"),("Cup_Open","None"),("Cup_Open","None"),
            ("Cup_Open","None"),
            ("None","Down"),("None","Down"),("None","Down"),("None","Down"),
            ("Cup_Closed","None"),("Cup_Closed","None"),("Cup_Closed","None"),
            ("Cup_Closed","None"),("Cup_Closed","None"),("Cup_Closed","None"),
            ("Cup_Closed","None")]

    Expected_TS = [("ThumbsUp","None"), ("None","Up"), ("Cup_Open","None"), ("None","Down"), ("Cup_Closed","None")]
    Expected_Gesture = 'Good Night'
    tools.eq_(classifyGestureByHMM(IP_Ts)[0],Expected_Gesture)

def test_noisy1_series_of_Good_Night():
    IP_Ts = [("ThumbsUp","None"),("None","Up"),("Cup_Closed","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),("ThumbsUp","None"),
            ("ThumbsUp","None"),("ThumbsUp","None"),
            ("None","Up"),("ThumbsUp","None"),
            ("None","Up"),("None","Up"),("None","Up"),("None","Up"),("None","Up"),
            ("Cup_Open","None"),("Cup_Closed","None"),("Cup_Open","None"),
            ("Cup_Open","None"),("Cup_Open","None"),("Cup_Open","None"),
            ("Cup_Closed","None"),
            ("None","Down"),("ThumbsUp","None"),
            ("None","Down"),("None","Down"),("None","Down"),("None","Down"),
            ("Cup_Closed","None"),("ThumbsUp","None"),("Cup_Closed","None"),
            ("Cup_Closed","None"),("ThumbsUp","None"),("Cup_Closed","None"),
            ("Cup_Closed","None")]

    Expected_TS = [("ThumbsUp","None"), ("None","Up"), ("Cup_Open","None"), ("None","Down"), ("Cup_Closed","None")]
    Expected_Gesture = 'Good Night'
    tools.eq_(classifyGestureByHMM(IP_Ts)[0],Expected_Gesture)
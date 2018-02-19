'''
* gesture_classify.py is the last stage of the gesture recognition mechanism. This file must be invoked by using the
* function recognize(sequence).
* This file holds all the automata required for various gestures.

* Version: 1.0
* Date: 20-2-2017
* Author: Varun
'''

def ATBRecognizer(sequence):
    state = 0
    finalStateReached = False
    for part in sequence:
        if state==0:
            if part[0]=="ThumbsUp":
                state = 1
                finalStateReached = True
                break
            else:
                return False
    return finalStateReached

def GMRecognizer(sequence):
    state = 0
    finalStateReached = False
    for part in sequence:
        if state==0:
            if part[0]=="ThumbsUp":
                state = 1
            else:
                return False
        if state == 1:
            if part[0]=="ThumbsUp":
                state = 1
            elif part[1]=="Down":
                state = 2
        if state == 2:
            if part[1]=="Down":
                state = 2
            elif part[0]=="Cup_Closed":
                state = 3
        if state == 3:
            if part[0]=="Cup_Closed":
                state = 3
            elif part[1]=="Up":
                state = 4
        if state == 4:
            if part[0]=="Cup_Open":
                state = 5
                finalStateReached = True
                break
            elif part[1]=="Up":
                state = 4
    return finalStateReached

def GNRecognizer(sequence):
    state = 0
    finalStateReached = False
    for part in sequence:
        if state==0:
            if part[0]=="ThumbsUp":
                state = 1
            else:
                return False
        if state == 1:
            if part[0]=="ThumbsUp":
                state = 1
            elif part[1]=="Up":
                state = 2
        if state == 2:
            if part[1]=="Up":
                state = 2
            elif part[0]=="Cup_Open":
                state = 3
        if state == 3:
            if part[0]=="Cup_Open":
                state = 3
            elif part[1]=="Down":
                state = 4
        if state == 4:
            if part[0]=="Cup_Closed":
                state = 5
                finalStateReached = True
                break
            elif part[1]=="Down":
                state = 4
    return finalStateReached

def GARecognizer(sequence):
    state = 0
    finalStateReached = False
    for part in sequence:
        if state == 0:
            if part[0]=="ThumbsUp":
                state = 1
            else:
                return False
        if state == 1:
            if part[0]=="ThumbsUp":
                state = 1
            elif part[1]=="Up":
                state = 2
        if state == 2:
            if part[1]=="Up":
                state = 2
            elif part[0]=="Sun_Up":
                state = 3
                finalStateReached = True
                break
    return finalStateReached

def recognize(sequence):
    '''
    * recognize(sequence)
    * This function will pass the gesture sequence to various automata for classification.
    * @param sequence: a list of tuples following the convention:
    * For Sign: (Sign,'None')
    * For Movement: ('None',Direction)
    * @return String which is the gesture.
    '''
    gesture=""
    gesture_recognized=False
    if ATBRecognizer(sequence):
        gesture_recognized=True
        gesture="All the best"
    if GMRecognizer(sequence):
        gesture_recognized=True
        return "Good Morning"
    if GARecognizer(sequence):
        gesture_recognized=True
        return "Good Afternoon"
    if GNRecognizer(sequence):
        gesture_recognized=True
        return "Good Night"
    if not gesture_recognized:
        gesture="Wrong Gesture"
    return gesture

def test_recognize_ATB():
    assert recognize([("ThumbsUp","None")]) == "All the best"
    assert recognize([("ThumbsUp","None"),("ThumbsUp","None"),("None","Up"),("ThumbsUp","None"),("None","Down")]) == "All the best"

def test_recognize_GM():
    assert recognize([("ThumbsUp","None"),("None","Down"),("None","Down"),("None","Down"),("Cup_Closed","None"),("None","Up"),("None","Up"),("Cup_Open","None"),("Cup_Open","None")]) == "Good Morning"
    assert recognize([("ThumbsUp","None"),("None","Down"),("Cup_Closed","None"),("None","Up"),("None","Up"),("Cup_Open","None"),("Cup_Open","None")]) == "Good Morning"
    assert recognize([("ThumbsUp","None"),("None","Down"),("None","Down"),("None","Down"),("Cup_Closed","None"),("None","Up"),("None","Up"),("Cup_Open","None"),("Cup_Open","None")]) == "Good Morning"

def test_recognize_GA():
    assert recognize([("ThumbsUp","None"),("None","Up"),("Sun_Up","None"),("Sun_Up","None"),("Sun_Up","None")]) == "Good Afternoon"
    assert recognize([("ThumbsUp","None"),("None","Up"),("None","Up"),("None","Up"),("Sun_Up","None")]) == "Good Afternoon"
    assert recognize([("ThumbsUp","None"),("ThumbsUp","None"),("ThumbsUp","None"),("None","Up"),("None","Up"),("None","Up"),("Sun_Up","None")]) == "Good Afternoon"

def test_recognize_Error():
    assert recognize([("None","Up"),("None","Up"),("None","Up"),("Sun_Up","None"),("ThumbsUp","None"),("Sun_Up","None"),("Sun_Up","None")]) == "Wrong Gesture"
    assert recognize([("None","Down"),("None","Down"),("None","Down"),("None","Down"),("ThumbsUp","None"),("None","Down"),("None","Down"),("Cup_Closed","None"),("None","Up"),("None","Up")]) == "Wrong Gesture"

def ATBRecognizer(sequence):
    state=0
    finalStateReached=False
    for part in sequence:
        if state==0 and part[0]=="None":
            state=0
        elif state==0 and part[0]=="ThumbsUp":
            state=1
            finalStateReached=True
        elif state==1:
            if part[0]=="ThumbsUp" or part[0]=="None":
                state=1
    return finalStateReached



def recognize(sequence):
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
    if NOT gesture_recognized:
        gesture="Wrong Gesture"
    return gesture

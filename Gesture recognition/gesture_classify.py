def ATBRecognizer(sequence):
    state = 0
    finalStateReached = False
    for part in sequence:
        if state==0 and part[0]=="None":
            state = 0
        elif state==0 and part[0]=="ThumbsUp":
            state = 1
            finalStateReached = True
        elif state==1:
            if part[0]=="ThumbsUp" or part[0]=="None":
                state = 1
    return finalStateReached

def GMRecognizer(sequence):
    state = 0
    finalStateReached = False
    for part in sequence:
        if state==0:
            if part[0]=="ThumbsUp":
                state = 1
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

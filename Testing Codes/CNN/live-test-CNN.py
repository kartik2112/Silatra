from keras.models import model_from_json
import cv2
import os
from random import randint
import numpy as np
import time

def load_model():
    print("Loading model\r",end="")
    json_file = open('CNN-model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("CNN-model.h5")
    print("Loaded model from disk.\r",end="")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model compiled and ready to use.")
    return model

saved_model=load_model()
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cv2.imshow('Smile!',frame)
    # Our operations on the frame come here
    image=cv2.resize(frame,(32,32))
    image=np.array(image)
    image = np.expand_dims(image, axis=0)
    class_probs=saved_model.predict(image)
    predicted_digit=class_probs.argmax()
    print("Predicted Digit:"+str(predicted_digit)+"\r")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

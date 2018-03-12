from keras.models import model_from_json
import cv2
import os
from random import randint
import numpy as np
import time

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
rand_tests_num=int(input("Enter number of random tests to run:"))
success_cases=0
total_time=0
lower = np.array([0,145,60],np.uint8)
upper = np.array([255,180,127],np.uint8)
for i in range(rand_tests_num):
    test_digit=chr(randint(97,122))
    while test_digit=='h' or test_digit=='j' or test_digit=='v':
        test_digit=chr(randint(97,122))
    print("Chosen digit:"+str(test_digit))
    print("Prediction in progress...\r",end="")
    folder="C:/Users/VR-Admin/Pictures/Training images/training-images-kartik/Letters/"+str(test_digit)
    file_number=randint(1,len(os.listdir(folder)))
    img_name=folder+"/"+str(file_number)+".png"
    start_time=time.time()
    image=cv2.imread(img_name)
    blur = cv2.blur(image,(3,3))
    ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)
    mask2 = cv2.inRange(ycrcb,lower,upper)
    image = cv2.bitwise_and(image,image,mask = mask2)
    image=cv2.resize(image,(50,50))
    image=np.array(image)
    image = np.expand_dims(image, axis=0)
    class_probs=model.predict(image)
    predicted_digit=chr(class_probs.argmax()+97)
    print("Predicted Digit:"+str(predicted_digit)+"            ")
    execution_time=time.time() - start_time
    if predicted_digit==test_digit:
        print("Model Success.")
        success_cases+=1
    else:
        print("Model Fail.")
    print("--- %s seconds ---" % (execution_time))
    total_time+=float(execution_time)
    print("\n")
print("Success Rate:{:.4f}".format((success_cases/rand_tests_num)*100))
print("Average Classification time:{:.4f} seconds".format((total_time/rand_tests_num)))

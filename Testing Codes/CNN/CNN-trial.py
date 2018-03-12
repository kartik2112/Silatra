import numpy as np
import pandas as pd
# import scipy
# from skimage import io
import cv2
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder

lower = np.array([0,145,60],np.uint8)
upper = np.array([255,180,127],np.uint8)
letters=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
images=[]
labels=[]
for letter in letters:
    letter_files=os.listdir("training-images-kartik/Letters/"+str(letter))
    for image_file in letter_files:
        image = cv2.imread("training-images-kartik/Letters/"+str(letter)+"/"+image_file)
        print("Reading Image:"+str(letter)+"/"+image_file+"\t\r", end="")
        blur = cv2.blur(image,(3,3))
        ycrcb = cv2.cvtColor(blur,cv2.COLOR_BGR2YCR_CB)
        mask2 = cv2.inRange(ycrcb,lower,upper)
        image = cv2.bitwise_and(image,image,mask = mask2)
        images.append(image)
        labels.append(ord(letter)-97)
print("Images read.")
for i in range(len(images)):
    images[i]=cv2.resize(images[i],(100,100))
# cv2.imshow('100x100 image',images[3])
# exit()
images = np.array(images)
labels = np_utils.to_categorical(labels)
# Segregating training and testind data:
train_data,test_data,train_labels,test_labels=train_test_split(images,labels,test_size=0.33)
# Model build:
model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(100,100,3)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(letters), activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Hope and Pray it works all well:
history=model.fit(train_data,train_labels,batch_size=256,epochs=15,verbose=1,validation_split=0.1)
# Plot a curve of the model performance:
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.show()

test_scores=model.evaluate(test_data,test_labels)
print("Accuracy on test set:{:.4f}".format(test_scores[1]*100))

print("Saving model\r",end="")
model_json = model.to_json()
with open("CNN-model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("CNN-model.h5")
print("Saved model to disk")

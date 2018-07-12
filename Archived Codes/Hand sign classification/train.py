from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.utils import np_utils
from sklearn.neighbors import KNeighborsClassifier

data, desired = [], []
min_x, max_x = 999, 0
min_y, max_y = min_x, max_x
with open('norm_data.csv') as f:
    f.readline() # Skip labels
    while True:
        line = f.readline()
        if line == '': break
        line = line.strip().split(',')
        line[:-1] = map(float,line[:-1])
        line[-1] = int(line[-1])-1
        data.append(line[:-1])
        desired.append(line[-1])


train_data, test_data, train_labels, test_labels = train_test_split(data,desired,test_size=0.33,random_state=101)

def deep():
    encoder = LabelEncoder()
    encoder.fit(train_labels)
    encoded_train_labels = encoder.transform(train_labels)
    dummy_train_labels = np_utils.to_categorical(encoded_train_labels)

    encoder = LabelEncoder()
    encoder.fit(test_labels)
    encoded_test_labels = encoder.transform(test_labels)
    dummy_test_labels = np_utils.to_categorical(encoded_test_labels)

    model = Sequential()
    model.add(Dense(40, input_dim=4,activation='relu',name='Hidden_layer_1'))
    model.add(Dense(80,activation='relu',name='Hidden_layer_2'))
    model.add(Dense(150,activation='relu',name='Hidden_layer_3'))
    model.add(Dense(300,activation='relu',name='Hidden_layer_4'))
    model.add(Dense(500,activation='relu',name='Hidden_layer_5'))
    model.add(Dense(2,activation='softmax',name='Output_layer'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data,dummy_train_labels,epochs=120,verbose=1,validation_split=0.1)

    score = model.evaluate(test_data,dummy_test_labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    to_be_saved_model = model.to_json()
    with open('model.json','w') as model_file: model_file.write(to_be_saved_model)
    model.save_weights('weights.h5')

def KNN():
    knn = KNeighborsClassifier(n_neighbors=2)

deep()
from sklearn.model_selection import train_test_split
from numpy import array, uint8
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import cv2

'''
HSV = Hue, Saturation & Value.
Illumination is reflected by V values. We are not interested in Illumination, we are interested in actual colour.abs
Actual colour is determined by HS values. Thus, we neglect V as it is irrelevant in training.
'''

data,d=[],[]
with open('data.txt') as f:
    while True:
        # Read data row by row
        line = f.readline()
        if line == '': break                    # End of file
        line = line.split('\t')
        pixel = line[0:len(line)-1]             # Data comes as: [b,g,r,class]. Select BGR values

        # Conversion of BGR to HS colour space.
        hsv_pixel = cv2.cvtColor(uint8([[pixel]]), cv2.COLOR_BGR2HSV).tolist()[0][0]
        hs_pixel = hsv_pixel[0:2]               # Neglect V from HSV
        pixel = hs_pixel

        # Append data
        for i in range(len(pixel)): pixel[i] = float(pixel[i])/255.0
        data.append(pixel)
        d.append(line[len(line)-1])

# Split data for training & testing. Ratio = 33%
train_data,test_data,train_labels,test_labels = train_test_split(data,d,test_size=0.3,random_state=31)

def knn():
    # Imports
    from sklearn.neighbors import KNeighborsClassifier

    neighbors = KNeighborsClassifier(n_neighbors=2)             # 2 clusters

    # Training starts
    neighbors.fit(train_data,train_labels)

    # Test with testing data.
    correct,r,g,b=0,[],[],[]
    for i in range(len(test_data)):
        if test_labels[i] == neighbors.predict([test_data[i]])[0]:
            correct += 1
            if test_labels[i] == 1:
                r.append(test_data[i][0])
                g.append(test_data[i][1])
                b.append(test_data[i][2])
    print("Accuracy =",correct,"/",len(test_data),"=",correct*1.0/len(test_data))

def deep():
    # Imports
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline
    from keras.utils import np_utils

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(train_labels)
    encoded_Y = encoder.transform(train_labels)

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_labels = np_utils.to_categorical(encoded_Y)

    encoder = LabelEncoder()
    encoder.fit(test_labels)
    encoded_Y = encoder.transform(test_labels)

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_test_labels = np_utils.to_categorical(encoded_Y)

    # This is the best random seed!
    np.random.seed(101)

    '''
    This is a sequential model.
    Contains 3 inputs,
    1 hidden layer with 8 neurons & activation as ReLU
    & 1 output layer with 2 neurons and activation Softmax
    '''

    model = Sequential()
    model.add(Dense(8,input_dim=2,activation='relu', name='hidden_layer'))
    model.add(Dense(2, activation='softmax', name='output_layer'))

    # Compile model & fit data to model.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data,dummy_labels,batch_size=16,epochs=20,verbose=1,validation_split=0.25)

    # Evaluate against test data
    score = model.evaluate(test_data,dummy_test_labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    # Save model architecture in json file & save weights in another file.
    to_be_saved_model = model.to_json()
    with open('skin.json','w') as model_file: model_file.write(to_be_saved_model)
    model.save_weights('skin.h5')

# Program starts here
if __name__ == "__main__":
    deep()

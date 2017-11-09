from sklearn.model_selection import train_test_split
from numpy import array, uint8
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import cv2

'''
HSV = Hue, Saturation & Value.
Illumination is reflected by V values. We are not interested in Illumination, we are interested in actual colour.abs
Actual colour is determined by HS values. Thus, we might neglect V as it seems irrelevant in training.
But, in that case dark & light colours (close to black & white) also get detected. Also colours like pink are detected.
Thus, including V might help model in learning the range of V.

Ranges for HSV:
H       [0,255]
S,V     [0,100]%
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
        pixel = cv2.cvtColor(uint8([[pixel]]), cv2.COLOR_BGR2HSV).tolist()[0][0]

        # Normalize & Append data
        ranges = [255.0,100.0,100.0]
        for i in range(len(pixel)): pixel[i] = float(pixel[i])/ranges[i]
        
        data.append(pixel)
        d.append(line[len(line)-1])

with open('hs_new.data') as f:
    while True:
        # Read data row by row
        row = f.readline()
        if row is '': break                     # End of file
        row = row.split('\t')
        pixel = row[0:len(row)-1]               # Data comes as [h,s,v,class]. Select HS values.

        # Normalisation of HS data
        ranges = [255.0,100.0,100.0]
        for i in range(len(pixel)): pixel[i] = float(pixel[i])/ranges[i]

        # Appending data
        data.append(pixel)
        d.append(row[len(row)-1])

# Split data for training & testing. Ratio = 33%
train_data,test_data,train_labels,test_labels = train_test_split(data,d,test_size=0.3,random_state=31)

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
    model.add(Dense(8,input_dim=3,activation='relu', name='hidden_layer'))
    model.add(Dense(2, activation='softmax', name='output_layer'))

    # Compile model & fit data to model.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data,dummy_labels,batch_size=16,epochs=20,verbose=1,validation_split=0.25)

    # Evaluate against test data
    score = model.evaluate(test_data,dummy_test_labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    # Save model architecture in json file & save weights in another file.
    to_be_saved_model = model.to_json()
    with open('new_skin_model.json','w') as model_file: model_file.write(to_be_saved_model)
    model.save_weights('new_skin_model_weights.h5')

# Program starts here
if __name__ == "__main__":
    deep()

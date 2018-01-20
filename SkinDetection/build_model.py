from sklearn.model_selection import train_test_split
from numpy import array, uint8
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import cv2
from sklearn.metrics import roc_curve, auc

'''
HSV = Hue, Saturation & Value.
Illumination is reflected by V values. We are not interested in Illumination, we are interested in actual colour.abs
Actual colour is determined by HS values. Thus, we might neglect V as it seems irrelevant in training.
But, in that case dark & light colours (close to black & white) also get detected. Also colours like pink are detected.
Thus, including V might help model in learning the range of V for skin colours.

Ranges for HSV:
H       [0,179]
S,V     [0,255]
'''

def read_uci_data(data,d):
    print('Reading the UCI dataset...\r',end='')
    with open('uci_skin_segmentation_data.txt') as f:
        while True:
            # Read data row by row
            line = f.readline()
            if line == '': break                    # End of file
            line = line.split('\t')
            pixel = line[0:len(line)-1]             # Data comes as: [h,s,v,class].

            # Conversion of BGR to HSV colour space.
            pixel = cv2.cvtColor(uint8([[pixel]]), cv2.COLOR_BGR2HSV).tolist()[0][0]

            # Normalize & Append data
            ranges = [179.0,255.0,255.0]
            for i in range(len(pixel)): pixel[i] = float(pixel[i])/ranges[i]

            data.append(pixel)
            d.append(line[len(line)-1])

def read_silatra_data(data,d):
    with open('data.txt') as f:
        row_count=1
        print('Reading the Silatra dataset... Read 0 Lakh rows\r',end='')
        while True:
            # Read data row by row
            row = f.readline()
            if row is '': break                     # End of file
            row = row.split('\t')
            pixel = row[0:len(row)-1]               # Data comes as [h,s,v,class]

            if row_count%100000 == 0: print('Reading the Silatra dataset... Read '+str(int(row_count/100000))+' Lakh rows\r',end='')
            row_count+=1

            # Normalisation of HSV data
            ranges = [179.0,255.0,255.0]
            for i in range(len(pixel)): pixel[i] = float(pixel[i])/ranges[i]

            # Appending data
            data.append(pixel)
            d.append(row[len(row)-1])


def deep(data,d):
    # Imports
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline
    from keras.utils import np_utils

    X_train=[]
    Y_train=[]
    input_file=open("skin-detection-training.txt","r")
    for line in input_file:
        attrs=line.split(",")
        pixel=list(map(float,attrs[0:3]))
        for i in range(3):pixel[i] /= ranges[i]
        Y_train.append(int(attrs[-1].strip())-1)
        X_train.append(pixel)
    print("Number of training samples loaded:"+str(len(X_train)))
    X_test=[]
    Y_test=[]
    input_file=open("skin-detection-testing.txt","r")
    for line in input_file:
        attrs=line.split(",")
        pixel=list(map(float,attrs[0:3]))
        for i in range(3):pixel[i] /= ranges[i]
        Y_test.append(int(attrs[-1].strip())-1)
        X_test.append(pixel)
    print("Number of test samples loaded:"+str(len(X_test)))

    np.random.seed = 101
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y_train)
    enc_Y_train = encoder.transform(Y_train)
    dummy_train_labels = np_utils.to_categorical(enc_Y_train)

    encoder = LabelEncoder()
    encoder.fit(Y_test)
    enc_Y_test = encoder.transform(Y_test)
    dummy_test_labels = np_utils.to_categorical(enc_Y_test)

    model = Sequential()
    model.add(Dense(3,input_dim=3,activation='relu', name='input_layer'))
    model.add(Dense(120,activation='relu', name='hidden_layer_1'))
    model.add(Dense(80,activation='relu', name='hidden_layer_2'))
    model.add(Dense(40,activation='relu', name='hidden_layer_3'))
    model.add(Dense(20,activation='relu', name='hidden_layer_4'))
    model.add(Dense(2, activation='softmax', name='output_layer'))

    # Compile model & fit data to model.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train,dummy_train_labels,batch_size=32,epochs=14,verbose=1,validation_split=0.1)

    # Evaluate against test data
    score = model.evaluate(X_test,dummy_test_labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    # Save model architecture in json file & save weights in another file.
    print('Saving model....\r',end='')
    to_be_saved_model = model.to_json()
    with open('model1.json','w') as model_file: model_file.write(to_be_saved_model)
    model.save_weights('weights1.h5')

    print('You may now segment an image!')


# Program starts here
if __name__ == "__main__":
    print('\n----- Silatra Deep Learning -----\n')

    data, d = [], []
    ranges=[179.0,255.0,255.0]
    # read_uci_data(data,d)
    # read_silatra_data(data,d)
    #
    # # Split data for training & testing. Ratio = 33%
    # train_data,test_data,train_labels,test_labels = train_test_split(data,d,test_size=0.3,random_state=31)
    print('Data is ready for training! Training starts... Hello Keras? You may have the control now..\nKeras: Sure.\n\n')
    deep(data,d)

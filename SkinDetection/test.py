from sklearn.model_selection import train_test_split
from numpy import array, uint8
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import cv2

data,d=[],[]
with open('data.txt') as f:
    while True:
        line = f.readline()
        if line == '': break
        line = line.split('\t')
        pixel = line[0:len(line)-1]

        # Conversion of BGR to HSV
        hsv_pixel = cv2.cvtColor(uint8([[pixel]]), cv2.COLOR_BGR2HSV).tolist()[0][0]
        hs_pixel = hsv_pixel[0:2]
        pixel = hs_pixel

        for i in range(len(pixel)): pixel[i] = float(pixel[i])/255.0
        data.append(pixel)
        d.append(line[len(line)-1])

train_data,test_data,train_labels,test_labels = train_test_split(data,d,test_size=0.3,random_state=31)

def knn():
    from sklearn.neighbors import KNeighborsClassifier
    neighbors = KNeighborsClassifier(n_neighbors=2)
    neighbors.fit(train_data,train_labels)
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
    # Imports needed
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
    1 hidden layer with 2 neurons & activation as sigmoid
    & 1 output layer with 1 neuron and activation relu
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

    r,g,b = 112,91,93
    data_bgr = [b,g,r]
    data_ycbcr = cv2.cvtColor(uint8([[data_bgr]]), cv2.COLOR_BGR2HSV).tolist()[0][0]
    data_to_test = data_ycbcr[1:]
    for i in range(len(data_to_test)): data_to_test[i] = data_to_test[i]*1.0/255.0
    data_to_test = [data_to_test]
    output = model.predict(array(data_to_test)).tolist()
    print('Output of model: '+str(output))
    class_dt=""
    if output[0][0]>output[0][1]:
        class_dt="Skin"
    else:
        class_dt="Non-Skin"
    print("Predicted class:"+class_dt)

# Program starts here
if __name__ == "__main__":
    deep()

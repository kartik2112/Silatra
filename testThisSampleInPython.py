# Reference for pickle:
#    https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/


import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pickle

def predictSignByKNN(line):
    noOfDescriptors = 10
    fftData=[]

    # print("Received from C++",line)

    data = np.fromstring(line,dtype = float, sep = ',')
    fftData.append(fft(data)[0:noOfDescriptors])  # FFT

    # print("Computed FFT")
    # print(fftData)

    fftData = np.absolute(fftData) 

    neigh = pickle.load(open('KNNModelDump.sav','rb'))
    print("Loaded KNN Model")

    return neigh.predict([ fftData[0] ])[0]

def predictSignByDeepNet(line):
    noOfDescriptors = 15
    fftData=[]

    print("Received from C++",line)

    data = np.fromstring(line,dtype = float, sep = ',')
    fftData.append(fft(data)[0:noOfDescriptors])  # FFT

    print("Computed FFT")
    print(fftData)
    
    fftData = np.absolute(fftData) 
    json_file = open('Classification Models/DigitClassifierModel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    # loaded_model.load_weights("MLModels/KerasModel.h5")
    loaded_model.load_weights("Classification Models/DigitClassifierModel.h5")
    print("Loaded model from disk")

    print("Compiling the loaded model.")
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    predicted_digit=loaded_model.predict(fftData)
    print(predicted_digit)
    for x in predicted_digit:
        print("The predicted digit is:"+str(np.argmax(x)))
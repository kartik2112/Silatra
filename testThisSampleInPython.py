# Reference for pickle:
#    https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/


import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.neighbors import KNeighborsClassifier
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
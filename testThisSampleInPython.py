import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.neighbors import KNeighborsClassifier

import pickle

noOfDescriptors = 10
fftData=[]

testSampleCSVFile = open("./TestSampleDistancesData.csv")
print("Reading test sample distances data")

#print(data)
ctr = 0
for line in testSampleCSVFile:
    data = np.fromstring(line,dtype = float, sep = ',')
    fftData.append(fft(data)[0:noOfDescriptors])  # FFT
print("Computed FFT")
print(fftData)

fftData = np.absolute(fftData) 

neigh = pickle.load(open('KNNModelDump.sav','rb'))
print("Loaded KNN Model")

print("Predicted Sign:" , neigh.predict([ fftData[0] ])[0] )
import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

from sklearn.svm import SVC
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
#For plotting parallel coordinates
# from pandas.plotting import parallel_coordinates
# import matplotlib.pyplot as plt
# import pandas as pd

# Initializers
dataInds = [1,2,3,4,5,6,7,8]
subFolderNames = ['Normal'] #,'Rotated'
noOfDescriptors = 10
noOfSamples = []
fftData=[]
# storeAsLabelledFeaturesFile = True
storeAsLabelledFeaturesFile = False

def normalise(x):
	'''
		Returns a numpy array normalised ALONG the COLUMNS.
		The new range is 0 to 1.
		This range can be changed by changing the variables new_min and new_max
	'''
	new_min,new_max=0,1
	# There's really no need to elaborate this code, its for dumbchucks like me :)
	min_along_columns=np.amin(x,axis=0)
	max_along_columns=np.amax(x,axis=0)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x[i][j]=(new_max-new_min)/(max_along_columns[j]-min_along_columns[j])*(x[i][j]-min_along_columns[j])+new_min
            # You can just put np.amin and np.amax straight in, but I put those seperately cause heyo dumbchucks remember :)

def duplicate(X,Y,per_sample=200):
	'''
		This function will increase the number of samples in the set by duplication.
	'''
	from random import randint
	dataops=[1,2,3,4,5,6,7,8]
	new_X_len=len(dataops)*per_sample
	new_X=np.zeros(shape=(new_X_len,10))
	new_Y=np.zeros(shape=(new_X_len,1))
	main_index=0
	for i in range(len(dataops)):
	    temp_data=[]
	    for j in range(len(Y)):
	        if Y[j]==dataops[i]:
	            temp_data.append(X[j])
	    used_samples=0
	    while used_samples<per_sample:
	        index_chosen=randint(0,len(temp_data)-1)
	        new_X[main_index]=temp_data[index_chosen]
	        new_Y[main_index]=dataops[i]
	        main_index+=1
	        used_samples+=1
	# print("Shape of X:"+str(new_X.shape))
	# print("Shape of Y:"+str(new_Y.shape))
	return new_X,new_Y


def dumpData():
	global fftData,correctLabels
	toBeDumpedData = []

	for i in range(len(fftData)):
		if storeAsLabelledFeaturesFile==True:
			toBeDumpedData.append(fftData[i].tolist() + [correctLabels[i]])      ##### Use this statement if you want to store the classes along with the descriptors data
		else:
			toBeDumpedData.append(fftData[i].tolist() + [correctLabels[i]])							 ##### Use this statement if you DO NOT want to store the classes along with the descriptors data

	if storeAsLabelledFeaturesFile==True:
		header_line = (','.join( str(x) for x in range(1,noOfDescriptors+1) )+",Class")
		np.savetxt("data.csv",toBeDumpedData, delimiter=",",header = header_line , comments = '' )
	else:
		np.savetxt("data.csv",toBeDumpedData, delimiter=",")
	print("Saved csv file")

def plotFeatures():
	plt.figure()
	data123 = pd.read_csv('data.csv')
	parallel_coordinates(data123, 'Class')
	plt.show()

def KMeansClustering():
	global noOfSamples,fftData,dataInds
	print("Applying KMeans clustering to data")
	kmeans = KMeans(n_clusters = len(noOfSamples), random_state = 0).fit(fftData)
	labels1 = kmeans.labels_

	labelsCluster = []
	offset = 0
	for i in range(0,len(noOfSamples)):
		labelsCluster.append(labels1[offset:offset+noOfSamples[i]].tolist())
		offset += noOfSamples[i]
	print(labelsCluster)
	for i in range(0,len(noOfSamples)):
		dict1 = {}
		for j in labelsCluster[i]:
			if j in dict1:
				dict1[j]+=1
			else:
				dict1[j]=1
		print(dataInds[i],":",dict1)

def KNearestNeighbors():
	global noOfSamples,fftData,dataInds,correctLabels
	print("Applying K Nearest Neighbours Learning to data")
	trainData_S,testData_S,trainData_L,testData_L = train_test_split(fftData,correctLabels,test_size = 0.33,random_state=42)

	neigh = KNeighborsClassifier(n_neighbors = len(noOfSamples))
	neigh.fit(trainData_S,trainData_L)

	correctlyClassified = 0
	for i in range(len(testData_S)):
		# print(testData_L[i],":",neigh.predict_proba([testData_S[i]]))
		print(testData_L[i],":",neigh.predict([testData_S[i]]))
		if( testData_L[i] == neigh.predict([testData_S[i]])[0] ):
			correctlyClassified += 1
	print("Accuracy: ",correctlyClassified,"/",len(testData_S),"=",correctlyClassified/len(testData_S))
	pickle.dump(neigh, open('KNNModelDump.sav','wb'))
	print("Model saved as 'KNNModelDump.sav'")

def SVMLearning():
	global noOfSamples,fftData,dataInds,correctLabels
	print("Applying SVM Learning to data")
	trainData_S,testData_S,trainData_L,testData_L = train_test_split(fftData,correctLabels,test_size = 0.33,random_state=42)

	svm=SVC(C=10,gamma=10)
	svm.fit(trainData_S,trainData_L)
	print("Accuracy on training set:"+str(svm.score(trainData_S,trainData_L)*100))
	print("Accuracy on training set:"+str(svm.score(testData_S,testData_L)*100))

def KerasDeepLearning():
	#Install Keras and Tensorflow/Theanos before using this function.
	X,Y=duplicate(fftData,correctLabels,600)
	train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.33,random_state=42)
	normalise(train_x)
	normalise(test_x)
	print("Training Set Size:"+str(train_x.shape[0]))
	print("Testing Set Size:"+str(test_x.shape[0]))
	# One-Hot encoding
	encoder = LabelEncoder()
	encoder.fit(train_y)
	encoded_train_Y = encoder.transform(train_y)
	dummy_train_y = np_utils.to_categorical(encoded_train_Y)
	encoder = LabelEncoder()
	encoder.fit(test_y)
	encoded_test_Y = encoder.transform(test_y)
	dummy_test_y = np_utils.to_categorical(encoded_test_Y)
	model = Sequential()
	model.add(Dense(10, input_dim=10, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(len(dataInds), activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(train_x,dummy_train_y,validation_split=0.1,epochs=150,batch_size=65,verbose=1)
	scores = model.evaluate(train_x,dummy_train_y)
	print("\n%s: %.2f%%" % ("Accuracy on Training set", scores[1]*100))
	scores = model.evaluate(test_x,dummy_test_y)
	print("\n%s: %.2f%%" % ("Accuracy on Testing set", scores[1]*100))
    # Next code is for saving the model to a JSON file:
  	# Model saving code:
  	# print("Saving Model.")
	# model_json = model.to_json()
	# with open("MLModels/KerasModel.json", "w") as json_file:
	# 	json_file.write(model_json)
	# model.save_weights("MLModels/KerasModel.h5")
	# print("Saved model to disk")

############# Main flow starts here #################

# Travers through csv files and append CCDC Data
for folderNo in dataInds:
	ctr = 0
	for subFolderNameI in subFolderNames:
		path_to_csv = "./CCDC-Data/training-images/Digits/"+str(folderNo)+"/Right_Hand/"+subFolderNameI+"/data.csv"

		#data = np.genfromtxt(path_to_csv, delimiter=',' )
		f1 = open(path_to_csv)

		#print(data)
		for line in f1:
			data = np.fromstring(line,dtype = float, sep = ',')
			fftData.append(fft(data)[0:noOfDescriptors])  # FFT
			ctr += 1
	noOfSamples.append(ctr)
	#print(fftData)
fftData = np.absolute(fftData)  # Making this rotation invariant by finding out magnitude
correctLabels = []
for i in range(len(noOfSamples)):
	correctLabels += [dataInds[i]]*noOfSamples[i]
# print(noOfSamples)
#
# print(fftData)

dumpData()

# KMeansClustering()
# KNearestNeighbors()
# SVMLearning()
KerasDeepLearning()

# plotFeatures()   # Keep this as the last statement if uncommented. Because this is a blocking operation
# Until you close the corresponding window created, program wont proceed any further.

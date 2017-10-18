import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

epochs_num=100
batch=128
verbose_stat=0

def dumpData():
	global fftData,correctLabels
	toBeDumpedData = []

	for i in range(len(fftData)):
		toBeDumpedData.append(fftData[i].tolist() + [correctLabels[i]])

	np.savetxt("data.csv",toBeDumpedData,delimiter=",")
	print("Saved csv file")

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

def SVMLearning():
	global noOfSamples,fftData,dataInds,correctLabels
	print("Applying SVM Learning to data")
	trainData_S,testData_S,trainData_L,testData_L = train_test_split(fftData,correctLabels,test_size = 0.33,random_state=42)

	svm=SVC(C=10,gamma=10)
	svm.fit(trainData_S,trainData_L)
	print("Accuracy on training set:"+str(svm.score(trainData_S,trainData_L)*100))
	print("Accuracy on training set:"+str(svm.score(testData_S,testData_L)*100))

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(64, input_dim=10, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(5, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def KerasDeepLearning():
	#Install Keras and Tensorflow/Theanos before using this function.
	seed=7
	np.random.seed(seed)
	#load the stuff
	dataframe = pd.read_csv("data.csv", header=None)
	dataset = dataframe.values
	X = dataset[:,0:10].astype(float)
	Y = dataset[:,10]
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	# convert integers to dummy variables (i.e. one hot encoded)
	dummy_y = np_utils.to_categorical(encoded_Y)
	estimator = KerasClassifier(build_fn=baseline_model, epochs=epochs_num, batch_size=batch, verbose=verbose_stat)
	print("Estimator created.")
	kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
	results = cross_val_score(estimator, X, dummy_y, cv=kfold)
	print("\nBaseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	print("Saving Model.")
	toBeSavedModel=baseline_model()
	toBeSavedModel.fit(X,dummy_y,epochs=epochs_num,batch_size=batch,verbose=1)
	model_json = toBeSavedModel.to_json()
	with open("MLModels/KerasModel.json", "w") as json_file:
		json_file.write(model_json)
	toBeSavedModel.save_weights("MLModels/KerasModel.h5")
	print("Saved model to disk")

# Initializers
dataInds = [1,2,3,4,5]
noOfDescriptors = 10
noOfSamples = []
fftData=[]

# Travers through csv files and append CCDC Data
for folderNo in dataInds:
	path_to_csv = "./CCDC-Data/training-images/Digits/"+str(folderNo)+"/Right_Hand/Normal/data.csv"

	#data = np.genfromtxt(path_to_csv, delimiter=',' )
	f1 = open(path_to_csv)

	#print(data)
	ctr = 0
	for line in f1:
		data = np.fromstring(line,dtype = float, sep = ',')
		fftData.append(fft(data))  # FFT
		ctr += 1
	noOfSamples.append(ctr)
	#print(fftData)
fftData = np.absolute(fftData)   # Making this rotation invariant by finding out magnitude
correctLabels = []
for i in range(len(noOfSamples)):
	correctLabels += [dataInds[i]]*noOfSamples[i]
print(noOfSamples)

print(fftData)

dumpData()

# KMeansClustering()
# KNearestNeighbors()
# SVMLearning()
# KerasDeepLearning()

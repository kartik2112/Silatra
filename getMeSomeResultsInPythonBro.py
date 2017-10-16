import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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
	print("Applying K Nearest Neighbouts Learning to data")
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
	
dataInds = [1,2,3,4,5]
noOfDescriptors = 10
noOfSamples = []
fftData=[]

for folderNo in dataInds:
	path_to_csv = "./CCDC-Data/training-images/Digits/"+str(folderNo)+"/Right_Hand/Normal/data.csv"

	#data = np.genfromtxt(path_to_csv, delimiter=',' )
	f1 = open(path_to_csv)

	#print(data)
	ctr = 0
	for line in f1:
		data = np.fromstring(line,dtype = float, sep = ',')
		fftData.append(fft(data)[:noOfDescriptors])
		ctr += 1
	noOfSamples.append(ctr)
	#print(fftData)
fftData = np.absolute(fftData)
correctLabels = []
for i in range(len(noOfSamples)):
	correctLabels += [dataInds[i]]*noOfSamples[i]
print(noOfSamples)

print(fftData)

toBeDumpedData = []

for i in range(len(fftData)):
	toBeDumpedData.append(fftData[i].tolist() + [correctLabels[i]])

np.savetxt("data.csv",toBeDumpedData,delimiter=",")
print("Saved csv file")

# KMeansClustering()
# KNearestNeighbors()

import pandas as pd
import numpy as np

dataFrame=pd.read_csv('totalData.csv')
# print(dataFrame.drop('target',axis=1))
# print(dataFrame['target'])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataFrame.drop('target',axis=1),dataFrame['target'],random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

print("k-NN model trained")
print("Test score:{:.2f}".format(knn.score(X_test,y_test)))

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)

print("SVM trained")
print("Test score:{:.2f}".format(svc.score(X_test,y_test)))

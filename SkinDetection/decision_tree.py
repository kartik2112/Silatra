# from __future__ import division  ---in case of using Python 2.x
import numpy
# Function for evaluating score of the tree
def evaluate_score(clf,X,Y):
    Y_predicted=list(map(int,(clf.predict(X)).tolist()))
    correct_predicts=0
    for i in range(0,len(Y)):
        if Y[i]==Y_predicted[i]:
            correct_predicts=correct_predicts+1
    acc=(correct_predicts/len(X))*100
    return acc
# Loading the file
X=[]
Y=[]
input_file=open("silatra_dataset_complete.txt","r")
for line in input_file:
    attrs=line.split("\t")
    Y.append(int(attrs[-1].strip()))
    X.append(list(map(int,attrs[0:3])))
print("Number of samples loaded:"+str(len(X)))
from random import randint
train_ratio=0.9
train_samples=int(train_ratio*len(X))
X_train=[]
Y_train=[]
while len(X_train)<train_samples:
    index=int(randint(0,len(X)-1))
    X_train.append(X[index])
    Y_train.append(Y[index])
    del X[index]
    del Y[index]
X_test=X
Y_test=Y
print("Number of Training samples:"+str(len(X_train)))
print("Number of Testing samples:"+str(len(X_test)))

# Decision Tree fitting:
from sklearn import tree
classifier=tree.DecisionTreeClassifier()
classifier.fit(X_train,Y_train)
train_score=evaluate_score(classifier,X_train,Y_train)
print("Accuracy on training set:"+str(train_score))
test_score=evaluate_score(classifier,X_test,Y_test)
print("Accuracy on test set:"+str(test_score))
# Trying the classifier on an image
import cv2,numpy as np,time
start = time.clock()
img=cv2.imread('Test_Images/varun.jpg')
img = cv2.resize(img,(320,240))
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = img.tolist()
seg_img=[]
for row in img:
    output=list(map(int,(classifier.predict(row)).tolist()))
    pixel_vals=[]
    for prediction in output:
        if prediction==1:
            pixel_vals.append([255.0,255.0,255.0])
        else:
            pixel_vals.append([0.0,0.0,0.0])
    seg_img.append(pixel_vals)
seg_img=np.array(seg_img)
end = time.clock()
print('Time required for segmentation: '+str(round(end-start,3))+'s')
cv2.imshow('Segmentated image',seg_img)
cv2.waitKey(100000)
cv2.destroyAllWindows()

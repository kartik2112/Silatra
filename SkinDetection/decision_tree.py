# from __future__ import division  ---in case of using Python 2.x
import numpy
ranges=[179.0,255.0,255.0]
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
X_train=[]
Y_train=[]
input_file=open("skin-detection-training.txt","r")
for line in input_file:
    attrs=line.split(",")
    Y_train.append(int(attrs[-1].strip()))
    X_train.append(list(map(float,attrs[0:3])))
print("Number of training samples loaded:"+str(len(X_train)))
X_test=[]
Y_test=[]
input_file=open("skin-detection-testing.txt","r")
for line in input_file:
    attrs=line.split(",")
    Y_test.append(int(attrs[-1].strip()))
    X_test.append(list(map(float,attrs[0:3])))
print("Number of test samples loaded:"+str(len(X_test)))

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
img=cv2.imread('Test_Images/good hand.jpg')
# if float(len(img)/len(img[0])) == float(16/9): img = cv2.resize(img, (180,320))
# elif float(len(img)/len(img[0])) == float(9/16): img = cv2.resize(img, (320,180))
# elif float(len(img)/len(img[0])) == float(4/3): img = cv2.resize(img, (320,240))
# elif float(len(img)/len(img[0])) == float(3/4): img = cv2.resize(img, (240,320))
# elif float(len(img)/len(img[0])) == 1: img = cv2.resize(img, (300,300))
# else: img = cv2.resize(img, (250,250))
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = img.tolist()
seg_img=[]
for row in img:
    for i in range(len(row)):
        for j in range(len(row[i])):
            row[i][j]=row[i][j]/ranges[j]
    output=list(map(int,(classifier.predict(row)).tolist()))
    pixel_vals=[]
    for prediction in output:
        if prediction==0:
            pixel_vals.append([255.0,255.0,255.0])
        else:
            pixel_vals.append([0.0,0.0,0.0])
    seg_img.append(pixel_vals)
seg_img=np.array(seg_img)
end = time.clock()
print('Time required for segmentation: '+str(round(end-start,3))+'s')
cv2.imshow('Segmentated image',seg_img)
cv2.imwrite("../Results and ROC/Decision Tree-BW.jpg",seg_img)
cv2.waitKey(100000)
cv2.destroyAllWindows()

# good hand.jpg: 1.367 s

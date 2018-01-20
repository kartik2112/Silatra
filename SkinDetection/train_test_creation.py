'''
    This python file will be used to create common training and testing sets for the decision trees, naive bayesian models and the deep neural network.
'''
'''
    Defining all the constants:
'''
input_file_name="silatra_dataset_complete.txt"
training_file_name="skin-detection-training.txt"
testing_file_name="skin-detection-testing.txt"
ranges=[179.0,255.0,255.0]
train_ratio=0.75

'''
    The following is the actual program:
'''
X=[]
Y=[]
input_file=open(input_file_name,"r")
for line in input_file:
    attrs=line.split("\t")
    Y.append(int(attrs[-1].strip())-1)
    X.append(list(map(int,attrs[0:3])))
print("Number of samples loaded:"+str(len(X)))
skin_samples=0
non_skin_samples=0
for label in Y:
    if label==0:
        skin_samples+=1
    else:
        non_skin_samples+=1
print("Skin samples in entire set:"+str(skin_samples))
print("Non-skin samples in entire set:"+str(non_skin_samples))
from random import randint
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
skin_samples=0
non_skin_samples=0
for label in Y_test:
    if label==0:
        skin_samples+=1
    else:
        non_skin_samples+=1
print("Skin samples in test set:"+str(skin_samples))
print("Non-skin samples in test set:"+str(non_skin_samples))

'''
    Writing the final sets onto files.
'''
print("Training samples:"+str(len(X_train)))
output_file_1=open(training_file_name,"w")
for i in range(len(X_train)):
    to_write=""
    for x in X_train[i]:
        to_write=to_write+str(x)+","
    to_write=to_write+str(Y_train[i])+"\n"
    output_file_1.write(to_write)
output_file_1.close()
print("Training samples put on file:"+training_file_name)

print("Testing samples:"+str(len(X_test)))
output_file_2=open(testing_file_name,"w")
for i in range(len(X_test)):
    to_write=""
    for x in X_test[i]:
        to_write=to_write+str(x)+","
    to_write=to_write+str(Y_test[i])+"\n"
    output_file_2.write(to_write)
output_file_2.close()
print("Training samples put on file:"+testing_file_name)

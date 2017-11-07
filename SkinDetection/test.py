from sklearn.model_selection import train_test_split
from numpy import array

data,d=[],[]
with open('data.txt') as f:
    while True:
        line = f.readline()
        if line == '': break
        line = line.split('\t')
        for i in range(len(line)-1): line[i] = float(line[i])/255.0
        data.append(line[0:len(line)-1])
        d.append(line[len(line)-1])

train_data,test_data,train_labels,test_labels = train_test_split(data,d,test_size=0.3,random_state=31)

def knn():
    from sklearn.neighbors import KNeighborsClassifier
    neighbors = KNeighborsClassifier(n_neighbors=2)
    neighbors.fit(train_data,train_labels)
    correct,r,g,b=0,[],[],[]
    for i in range(len(test_data)):
        if test_labels[i] == neighbors.predict([test_data[i]])[0]:
            correct += 1
            if test_labels[i] == 1:
                r.append(test_data[i][0])
                g.append(test_data[i][1])
                b.append(test_data[i][2])
    print "Accuracy =",correct,"/",len(test_data),"=",correct*1.0/len(test_data)

def deep():
    # Imports needed
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense

    # This is the best random seed!
    np.random.seed(101)

    ''' 
    This is a sequential model. 
    Contains 3 inputs, 
    1 hidden layer with 2 neurons & activation as sigmoid 
    & 1 output layer with 1 neuron and activation relu
    '''

    model = Sequential()
    model.add(Dense(2,input_dim=3,activation='sigmoid', name='hidden_layer'))
    model.add(Dense(1, activation='relu', name='output_layer'))

    # Compile model & fit data to model.
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    model.fit(train_data,train_labels,batch_size=32,epochs=20)

    # Evaluate against test data
    score = model.evaluate(test_data,test_labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    # Save model architecture in json file & save weights in another file.
    to_be_saved_model = model.to_json()
    with open('skin.json','w') as model_file: model_file.write(to_be_saved_model)
    model.save_weights('skin.h5')

    r,g,b = 203.0,213.0,253.0
    r,g,b = r/255.0, g/255.0, b/255.0
    data_to_test = [[r,g,b]]
    data_to_test = array(data_to_test)
    output = model.predict([data_to_test])
    print 'Input: (',r,',',g,',',b,') output:',output[0][0]
    
# Program starts here
if __name__ == "__main__":
    deep()
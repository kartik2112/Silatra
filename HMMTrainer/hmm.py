import numpy as np
import timeit
from hmmlearn import hmm
from sklearn.externals import joblib


np.random.seed(42)

no_of_its = 10000

Models = {
    'Good Afternoon': hmm.MultinomialHMM(n_components = 3,init_params='e',n_iter=no_of_its),
    'Good Morning': hmm.MultinomialHMM(n_components = 5,init_params='e',n_iter=no_of_its),
    'Good Night': hmm.MultinomialHMM(n_components = 5,init_params='e',n_iter=no_of_its)
}

Models['Good Afternoon'].startprob_ = np.array([1,0,0])
Models['Good Afternoon'].transmat_ = np.array([[0.5, 0.5, 0.0],
                           [0.0, 0.5, 0.5],
                           [0.0, 0.0, 1.0]])
#           Up, Right, Left, Down, Thumbs_Up, SunUp, Cup_Open, Cup_Closed
# Thumbs_Up 0.35,  0.025,  0.025, 0.025, 0.7, 0.11,  0.04,     0.04
# Up        0.7,0.08,  0.08, 0.01, 0.06,      0.06,  0.005,    0.005
# Done      0.0275,0.0275,0.0275,0.0275,0.11, 0.7,  0.04,     0.04

Models['Good Afternoon'].emissionprob_ = np.array([[0.35,  0.025,  0.025, 0.025, 0.7, 0.04,  0.04,     0.11],
                                [0.7,0.08,  0.08, 0.01, 0.06,      0.06,  0.005,    0.005],
                                [0.0275,0.0275,0.0275,0.0275,0.11, 0.7,  0.04,     0.04]])


Models['Good Night'].startprob_ = np.array([1,0,0,0,0])
Models['Good Night'].transmat_ = np.array([[0.5, 0.5, 0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5, 0.0, 0.0],
                           [0.0, 0.0, 0.5, 0.5, 0.0],
                           [0.0, 0.0, 0.0, 0.5, 0.5],
                           [0.0, 0.0, 0.0, 0.0, 1.0]])
#             Up,  Right,  Left,  Down,  Thumbs_Up, SunUp, Cup_Open, Cup_Closed
# Thumbs_Up   0.05,0.02,   0.02,  0.02,  0.7,       0.04,  0.04,     0.11
# Up          0.7, 0.08,   0.08,  0.01,  0.06,      0.06,  0.005,    0.005
# Cup_Open    0.05,0.005,  0.005, 0.05,  0.04,      0.11,  0.7,      0.04
# Down        0.01,0.08,   0.08,  0.7,   0.005,     0.005, 0.06,     0.06
# Cup_Closed  0.02,0.02,   0.02,  0.05,  0.11,      0.04,  0.04,     0.7

Models['Good Night'].emissionprob_ = np.array([[0.05,0.02,   0.02,  0.02,  0.7,       0.04,  0.04,     0.11],
                                [0.7, 0.08,   0.08,  0.01,  0.06,      0.06,  0.005,    0.005],
                                [0.05,0.005,  0.005, 0.05,  0.04,      0.11,  0.7,      0.04],
                                [0.01,0.08,   0.08,  0.7,   0.005,     0.005, 0.06,     0.06],
                                [0.02,0.02,   0.02,  0.05,  0.11,      0.04,  0.04,     0.7]])

Models['Good Morning'].startprob_ = np.array([1,0,0,0,0])
Models['Good Morning'].transmat_ = np.array([[0.5, 0.5, 0.0, 0.0, 0.0],
                           [0.0, 0.5, 0.5, 0.0, 0.0],
                           [0.0, 0.0, 0.5, 0.5, 0.0],
                           [0.0, 0.0, 0.0, 0.5, 0.5],
                           [0.0, 0.0, 0.0, 0.0, 1.0]])
#             Up,  Right,  Left,  Down,  Thumbs_Up, SunUp, Cup_Open, Cup_Closed
# Thumbs_Up   0.02,0.02,   0.02,  0.05,  0.7,       0.04,  0.04,     0.11
# Down        0.01,0.08,   0.08,  0.7,   0.06,      0.005, 0.005,    0.06
# Cup_Closed  0.05,0.005,  0.005, 0.05,  0.11,      0.04,  0.04,     0.7
# Up          0.7, 0.08,   0.08,  0.01,  0.005,     0.005, 0.06,     0.06
# Cup_Open    0.05,0.02,   0.02,  0.02,  0.04,      0.11,  0.7,      0.04

Models['Good Morning'].emissionprob_ = np.array([[0.02,0.02,   0.02,  0.05,  0.7,       0.04,  0.04,     0.11],
                                [0.01,0.08,   0.08,  0.7,   0.06,      0.005, 0.005,    0.06],
                                [0.05,0.005,  0.005, 0.05,  0.11,      0.04,  0.04,     0.7],
                                [0.7, 0.08,   0.08,  0.01,  0.005,     0.005, 0.06,     0.06],
                                [0.05,0.02,   0.02,  0.02,  0.04,      0.11,  0.7,      0.04]])



# train1 = np.array([[4],[4],[4],[4],[4],[0],[0],[0],[0],[0],[0],[0],[0],[0],[5],[5],[5],[5],[5],[5],[5],[5]])
# train2 = np.array([[4],[4],[4],[0],[4],[0],[0],[0],[0],[1],[0],[0],[2],[0],[0],[5],[5],[4],[5],[5],[4],[5],[5],[5],[5]])
# train3 = np.array([[4],[4],[4],[4],[1],[0],[0],[0],[0],[2],[0],[0],[0],[0],[5],[5],[5],[6],[5],[5],[5],[5]])
# train4 = np.array([[4],[4],[4],[0],[4],[0],[0],[0],[0],[0],[0],[0],[5],[0],[0],[5],[5],[0],[5],[5],[4],[5],[5],[5],[5]])
# train5 = np.array([[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[0],[0],[0],[0],[0],[0],[0],[0],[0],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5]])
# train6 = np.array([[4],[4],[4],[4],[4],[7],[7],[4],[4],[4],[4],[7],[4],[7],[4],[0],[0],[0],[0],[0],[0],[0],[0],[0],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5]])
# train7 = np.array([[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5]])
# train8 = np.array([[4],[4],[5],[0],[4],[0],[0],[0],[0],[1],[3],[0],[2],[0],[0],[5],[5],[4],[5],[5],[4],[5],[5],[5],[5]])
# train9 = np.array([[0],[0],[0],[0],[4],[4],[4],[0],[0],[0],[0],[0],[2],[0],[0],[0],[0],[5],[5],[5],[6],[5],[5],[5],[5]])
# trainData = np.concatenate([train1,train2,train3,train4,train5,train6,train7,train8,train9])
# lengths = [len(train1),len(train2),len(train3),len(train4),len(train5),len(train6),len(train7),len(train8),len(train9)]

GATrain1 = np.array([[4],[4],[7],[4],[4],[0],[0],[0],[0],[2],[0],[0],[0],[0],[1],[0],[0],[3],[0],[0],[5],[5],[5],[5],[5],[6],[5],[5],[5],[5],[5],[5]])
GATrain2 = np.array([[4], [0], [7], [4], [4], [4], [4], [4], [0], [0], [0], [0], [0], [0], [0], [5], [7], [6], [5], [5], [5], [5]])
GATrain3 = np.array([[4], [0], [7], [4], [4], [4], [4], [4], [0], [4], [2], [0], [0], [5], [5], [5], [5], [5], [6], [7]])

GMTrain1 = np.array([[4],[4],[4],[4],[4],[0],[4],[0],[4],[3],[3],[3],[3],[1],[3],[3],[3],[2],[3],[3],[7],[4],[7],[7],[7],[4],[7],[7],[7],[7],[7],[0],[0],[0],[0],[0],[0],[1],[0],[0],[0],[0],[0],[0],[2],[0],[6],[6],[6],[5],[6],[6],[6],[6],[6],[6],[5],[6],[6],[6]])
GMTrain2 = np.array([[4], [0], [7], [4], [4], [4], [4], [4], [3], [4], [3], [3], [3], [3], [3], [3], [3], [7], [4], [7], [7], [7], [7], [4], [7], [7], [7], [7], [0], [4], [0], [0], [0], [0], [0], [0], [0], [0], [6], [7], [6], [6], [6], [6], [7], [6], [6], [6], [7]])

trainData = {
    'Good Afternoon':np.concatenate([GATrain1, GATrain2, GATrain3]),
    'Good Morning':np.concatenate([GMTrain1, GMTrain2]),
    'Good Night':np.array([[4],[4],[4],[4],[4],[0],[4],[0],[4],[0],[0],[0],[0],[0],[0],[1],[0],[0],[0],[0],[0],[0],[2],[0],[6],[6],[6],[5],[6],[6],[6],[5],[6],[6],[6],[6],[6],[6],[3],[3],[3],[3],[1],[3],[3],[3],[2],[3],[3],[7],[7],[7],[4],[7],[7],[7],[7],[4],[7],[7]])
}

lengths ={
    'Good Afternoon':[len(GATrain1),len(GATrain2),len(GATrain3)],
    'Good Morning':[len(GMTrain1),len(GMTrain2)],
    'Good Night':[len(trainData['Good Night'])]
}


with open('gestures.csv') as fileR:
    while True:
        line = fileR.readline()
        if line == '': break
        line = line.strip().split(',')
        class1 = line[0]
        line[1:] = map(int,line[1:])
        data = line[1:]
        trainTemp = np.reshape(np.array(data),(-1,1))
        trainData[class1] = np.concatenate([trainData[class1],trainTemp])
        lengths[class1] += [len(trainTemp)]
        
print(trainData)
print(len(trainData))
print(lengths)
        

# joblib.dump(remodel, "filename.pkl")


for key1 in Models.keys():
    Models[key1].fit(trainData[key1],lengths[key1])
    print(Models[key1].startprob_)
    print(Models[key1].transmat_)
    print(Models[key1].emissionprob_)
    joblib.dump(Models[key1], "../Sign recognition/Models/GestureHMMs/"+key1+".pkl")
    print("Stored model '"+key1+".pkl' to path ''")

print()

test1 = np.array([[4],[4],[4],[4],[4],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5]])
test2 = np.array([[4],[4],[4],[4],[4],[0],[4],[0],[4],[0],[0],[0],[0],[0],[0],[0],[1],[0],[0],[2],[0],[0],[0],[0],[0],[1],[0],[0],[5],[5],[4],[5],[5],[4],[5],[5],[5],[5],[5],[5],[5],[5],[5]])
test3 = np.array([[4],[4],[4],[4],[4],[0],[4],[0],[4],[0],[0],[0],[0],[0],[0],[0],[1],[0],[6],[6],[6],[6],[6],[6],[6],[3],[3],[3],[3],[3],[3],[3],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7]])
test4 = np.array([[4],[4],[4],[4],[4],[0],[4],[0],[4],[3],[3],[3],[3],[3],[3],[3],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[0],[0],[0],[0],[0],[0],[0],[1],[0],[6],[6],[6],[6],[6],[6],[6]])

for key1 in Models.keys():
    print(key1)
    start_time = timeit.default_timer()
    print('Good Afternoon',Models[key1].score(test1))
    elapsed1 = timeit.default_timer() - start_time
    start_time = timeit.default_timer()
    print('Good Afternoon',Models[key1].score(test2))
    elapsed2 = timeit.default_timer() - start_time
    start_time = timeit.default_timer()
    print('Good Night',Models[key1].score(test3))
    elapsed3 = timeit.default_timer() - start_time
    start_time = timeit.default_timer()
    print('Good Morning',Models[key1].score(test4))
    elapsed4 = timeit.default_timer() - start_time
    
    print("Average time taken for this model",(elapsed1+elapsed2+elapsed3+elapsed4)/4)







# print(Models['Good Night'].sample(30))

# test1 = np.array([[4],[4],[4],[4],[4],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5]])
# test2 = np.array([[4],[4],[4],[4],[4],[0],[4],[0],[4],[0],[0],[0],[0],[0],[0],[0],[1],[0],[0],[2],[0],[0],[0],[0],[0],[1],[0],[0],[5],[5],[4],[5],[5],[4],[5],[5],[5],[5],[5],[5],[5],[5],[5]])
# test3 = np.array([[4],[4],[4],[4],[4],[0],[4],[0],[4],[0],[0],[0],[0],[0],[0],[0],[1],[0],[6],[6],[6],[6],[6],[6],[6],[3],[3],[3],[3],[3],[3],[3],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7],[7]])
# print(Models['Good Night'].score(test1))
# print(Models['Good Night'].score(test2))
# print(Models['Good Night'].score(test3))

# train1 = np.array([[4],[4],[4],[4],[4],[0],[0],[0],[0],[0],[0],[0],[0],[0],[5],[5],[5],[5],[5],[5],[5],[5]])
# train2 = np.array([[4],[4],[4],[0],[4],[0],[0],[0],[0],[1],[0],[0],[2],[0],[0],[5],[5],[4],[5],[5],[4],[5],[5],[5],[5]])
# train3 = np.array([[4],[4],[4],[4],[1],[0],[0],[0],[0],[2],[0],[0],[0],[0],[5],[5],[5],[6],[5],[5],[5],[5]])
# train4 = np.array([[4],[4],[4],[0],[4],[0],[0],[0],[0],[0],[0],[0],[5],[0],[0],[5],[5],[0],[5],[5],[4],[5],[5],[5],[5]])
# train5 = np.array([[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[0],[0],[0],[0],[0],[0],[0],[0],[0],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5]])
# train6 = np.array([[4],[4],[4],[4],[4],[7],[7],[4],[4],[4],[4],[7],[4],[7],[4],[0],[0],[0],[0],[0],[0],[0],[0],[0],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5]])
# train7 = np.array([[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[4],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5],[5]])
# train8 = np.array([[4],[4],[5],[0],[4],[0],[0],[0],[0],[1],[3],[0],[2],[0],[0],[5],[5],[4],[5],[5],[4],[5],[5],[5],[5]])
# train9 = np.array([[0],[0],[0],[0],[4],[4],[4],[0],[0],[0],[0],[0],[2],[0],[0],[0],[0],[5],[5],[5],[6],[5],[5],[5],[5]])
# X = np.concatenate([train1,train2,train3,train4,train5,train6,train7,train8,train9])
# lengths = [len(train1),len(train2),len(train3),len(train4),len(train5),len(train6),len(train7),len(train8),len(train9)]

# Models['Good Night'].fit(X,lengths)

# # print(Models['Good Night'].monitor)
# print(Models['Good Night'].startprob_)
# print(Models['Good Night'].transmat_)
# print(Models['Good Night'].emissionprob_)

# print()

# print(Models['Good Night'].score(test1))
# print(Models['Good Night'].score(test2))
# print(Models['Good Night'].score(test3))
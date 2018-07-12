import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle

data = pd.read_csv('new_data.csv')

X = data[['f'+str(i) for i in range(400)]].values
Y = data['label'].values

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, Y)

pickle.dump(classifier, open('KNN_Grid_ModelDump.sav','wb'))
print("Model saved as 'KNN_Grid_ModelDump.sav'")
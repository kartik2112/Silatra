from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pandas as pd
import numpy as np


# Loading the data
dataframe = pd.read_csv("TestSampleDistancesData.csv", header=None)
# dataframe = pd.read_csv("TestSampleDistancesData.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:10].astype(float)
class_array=["1","2","3","4","5","6","7","8","9"]
#For Keras Model
print("Getting the saved model.")
# json_file = open('MLModels/KerasModel.json', 'r')
json_file = open('Classification Models/DigitClassifierModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
# loaded_model.load_weights("MLModels/KerasModel.h5")
loaded_model.load_weights("Classification Models/DigitClassifierModel.h5")
print("Loaded model from disk")

print("Compiling the loaded model.")
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
predicted_digit=loaded_model.predict(X)
for x in predicted_digit:
    print("The predicted digit is:"+class_array[np.argmax(x)])

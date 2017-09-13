import numpy as np
from scipy.fftpack import fft, ifft

path_to_csv = "./CCDC-Data/training-images/Digits/2/Right_Hand/Normal/data.csv"

#data = np.genfromtxt(path_to_csv, delimiter=',' )
f1 = open(path_to_csv)

#print(data)

for line in f1:
	data = np.fromstring(line,dtype = float, sep = ',')
	print(fft(data)[:10])

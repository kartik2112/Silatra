import pandas as pd
import numpy as np, matplotlib.pyplot as plt
import matplotlib.cm as cm

data = pd.read_csv('Hand sign classification/silatra_signs_10x10.csv',dtype={100: np.unicode_})
print('Total data parsed: %d'%(len(data)))

X = data[['f'+str(i) for i in range(100)]].values
Y = data['label'].values

from sklearn.decomposition import PCA

pca = PCA(n_components=40)
result_after_pca = pca.fit_transform(X)

from sklearn.manifold import TSNE
import time

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=400).fit_transform(result_after_pca)
print( 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

no_of_classes = 36
label_indexes = {}
for i in range(26): label_indexes[chr(ord('a')+i)] = i
for i in range(10): label_indexes[str(i)] = 26+i

data_x = tsne[:,0]
data_y = tsne[:,1]

x, y, labels = [], [], []

for i in range(no_of_classes):
    x.append([])
    y.append([])
    labels.append('-')

print(len(data_x))
print(len(data_y))
print(len(Y))

for i in range(len(Y)):
    j = label_indexes[Y[i]]
    x[j].append(data_x[i])
    y[j].append(data_y[i])
    labels[j] = Y[i]

xs = np.arange(10)
ys = [i+xs+(i*xs)**2 for i in range(no_of_classes)]
colors = iter(cm.rainbow(np.linspace(0, 1, len(ys))))

for i in range(1,no_of_classes+1):
    plt.scatter(x[i-1], y[i-1], s=2, cmap=plt.cm.get_cmap("jet", no_of_classes), label=labels[i-1])

plt.title('ISL Static signs data extracted using a 10x10 grid')
plt.legend()
plt.show()
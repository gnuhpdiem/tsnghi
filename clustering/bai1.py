import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# read data
data = pd.read_csv('./data.csv', delimiter=' ', header=0)
data = data.to_numpy()

# k-means
k_means = KMeans(n_clusters=4)
k_means.fit(data)

# plot
plt.scatter(data[:, 0], data[:, 1], c=k_means.labels_)
plt.show()






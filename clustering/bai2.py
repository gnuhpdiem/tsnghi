from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # change to 2D

iris = load_iris()

data = iris.data

# to 2D
pca = PCA(n_components=2)
new_data = pca.fit_transform(data)

k_means = KMeans(n_clusters=3)
k_means.fit(new_data)

# plot
plt.scatter(new_data[:, 0], new_data[:, 1], c=k_means.labels_)
plt.show()
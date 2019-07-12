import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

data = pd.read_csv("./Advertising.csv", index_col=0)

feature_names = ['TV', 'radio', 'newspaper']

X = data[feature_names]
print(X)
print(X.size)
#kmeans = KMeans(n_clusters=3)
kmeans = KMeans().fit(X)
#kmeans = kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(labels)
print()
print(centroids)


colors = ["g.","r.","c.","y."]

plt.scatter(centroids[:, 0],centroids[:, 1], centroids[:, 2],marker = "x", linewidths = 10, zorder = 10)

plt.show()

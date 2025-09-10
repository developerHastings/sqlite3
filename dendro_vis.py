import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Perform hierarchical/agglomerative clustering using 'ward' linkage (minimizes variance within clusters)
linked = linkage(X, method='ward')

# Plot the dendrogram to visualize hierarchical merging
plt.figure(figsize=(12, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Extract cluster labels by cutting dendrogram at a specific height
max_d = 7  # max distance at which to cut for clusters
clusters = fcluster(linked, max_d, criterion='distance')

# Print cluster assignments for each sample
print("Cluster assignments for samples:")
print(clusters)


# Step 1: Import libraries
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Step 2: Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: Compute linkage matrix using Ward's method
Z = linkage(X, method='ward')

# Step 4: Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=12.)
plt.title('Hierarchical Clustering Dendrogram (Ward linkage)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Step 5: Cut the dendrogram to form clusters (3 clusters for Iris)
cluster_labels = fcluster(Z, t=3, criterion='maxclust')

# Step 6: Print cluster assignments
print("Cluster assignments for each sample:")
print(cluster_labels)

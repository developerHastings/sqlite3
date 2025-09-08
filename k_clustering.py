import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the iris dataset (ignoring the labels for unsupervised learning)
iris = load_iris()
X = iris.data # Features only (Sepal length, Sepal Width, Petal length, Petal width)

# Initialize KMeans clustering model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model to the data and predict cluster assignments
clusters = kmeans.fit_predict(X)

# Print cluster centers (centroids)
print("Cluster Centers (Centroids):")
print(kmeans.cluster_centers_)

# Calculate inertia (sum of squared distances to closest cluster center)
print(f"Inertia: {kmeans.inertia_:.2f}")

# Calculate and print silhouette score as a cluster quality metric
sil_score = silhouette_score(X, clusters)
print(f"Silhouette Score: {sil_score:.2f}")

# Plotting clusters and centroids using first two features for visualization
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('K-Means Clustering on Iris Dataset (Sepal Length vs Sepal Width)')
plt.legend()
plt.show()


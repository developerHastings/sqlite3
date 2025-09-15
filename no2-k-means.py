# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Step 3: Apply K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Step 4: Print cluster centers and inertia
print("Cluster centers:")
print(kmeans.cluster_centers_)
print("\nInertia (sum of squared distances):", kmeans.inertia_)

# Step 5: Calculate and print silhouette score
labels = kmeans.labels_
score = silhouette_score(X, labels)
print("\nSilhouette Score:", score)

# Step 6: Visualize clusters using first two features
plt.figure(figsize=(8,6))
colors = ['r', 'g', 'b']
for i in range(3):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], 
                color=colors[i], label=f'Cluster {i}', alpha=0.6)

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            color='yellow', marker='X', s=200, label='Centroids')

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('K-Means Clusters (Iris Dataset)')
plt.legend()
plt.show()

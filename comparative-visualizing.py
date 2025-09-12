# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# -----------------------------
# Step 3: K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
k_labels = kmeans.labels_
k_centers = kmeans.cluster_centers_

# -----------------------------
# Step 4: PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# -----------------------------
# Step 5: Hierarchical clustering
Z = linkage(X, method='ward')
from scipy.cluster.hierarchy import fcluster
h_labels = fcluster(Z, t=3, criterion='maxclust')

# -----------------------------
# Step 6: Comparative visualization (2x2 subplots)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ---- K-Means scatter plot ----
colors = ['r', 'g', 'b']
for i in range(3):
    axes[0,0].scatter(X[k_labels == i, 0], X[k_labels == i, 1], 
                      color=colors[i], label=f'Cluster {i}', alpha=0.6)
axes[0,0].scatter(k_centers[:, 0], k_centers[:, 1], color='yellow', marker='X', s=200, label='Centroids')
axes[0,0].set_title('K-Means Clusters (Features 1 & 2)')
axes[0,0].set_xlabel(feature_names[0])
axes[0,0].set_ylabel(feature_names[1])
axes[0,0].legend()

# ---- PCA projection colored by true labels ----
for i, target_name in enumerate(target_names):
    axes[0,1].scatter(X_pca[y==i,0], X_pca[y==i,1], label=target_name, alpha=0.7)
axes[0,1].set_title('PCA Projection (True Labels)')
axes[0,1].set_xlabel('PC1')
axes[0,1].set_ylabel('PC2')
axes[0,1].legend()

# ---- Dendrogram ----
dendro_ax = axes[1,0]
dendrogram(Z, ax=dendro_ax, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10.)
dendro_ax.set_title('Hierarchical Clustering Dendrogram')
dendro_ax.set_xlabel('Sample index')
dendro_ax.set_ylabel('Distance')

# ---- Hierarchical clustering flat cluster assignments ----
for i in range(3):
    axes[1,1].scatter(X[h_labels == i+1,0], X[h_labels == i+1,1], color=colors[i], label=f'Cluster {i+1}', alpha=0.6)
axes[1,1].set_title('Hierarchical Clustering (Ward, 3 clusters)')
axes[1,1].set_xlabel(feature_names[0])
axes[1,1].set_ylabel(feature_names[1])
axes[1,1].legend()

plt.tight_layout()
plt.show()

# -----------------------------
# Bonus Challenge: Elbow & Silhouette Analysis for K-Means
Ks = range(2, 10)
inertia_list = []
silhouette_list = []

for k in Ks:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia_list.append(km.inertia_)
    silhouette_list.append(silhouette_score(X, km.labels_))

# Plot Elbow and Silhouette
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(Ks, inertia_list, 'o-', color='blue')
plt.title('Elbow Plot (K-Means)')
plt.xlabel('Number of clusters K')
plt.ylabel('Inertia')

plt.subplot(1,2,2)
plt.plot(Ks, silhouette_list, 'o-', color='green')
plt.title('Silhouette Scores (K-Means)')
plt.xlabel('Number of clusters K')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# -----------------------------
# Bonus Challenge: Hierarchical linkage comparison
linkage_methods = ['single', 'complete', 'average', 'ward']
for method in linkage_methods:
    Z_temp = linkage(X, method=method)
    labels_temp = fcluster(Z_temp, t=3, criterion='maxclust')
    score = silhouette_score(X, labels_temp)
    print(f'Linkage: {method}, Silhouette Score: {score:.3f}')

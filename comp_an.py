import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load the Iris flower dataset
iris = load_iris()
X = iris.data 
y = iris.target  
feature_names = iris.feature_names
target_names = iris.target_names

# Create a PCA model to reduce to 2 principal components
pca = PCA(n_components=2)

# Fit PCA on the iris data and transform it
X_pca = pca.fit_transform(X)

# Show explained variance ratio per principal component
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Plot the first two principal components
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, lw=2, label=target_name)
plt.legend()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.show()



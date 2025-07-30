import numpy as np
import matplotlib.pyplot as plt


# Linear Algebra Operations with NumPy

# Check Columns are linearly independent
A = np.array([[1, 2], [3, 4]])
rank = np.linalg.matrix_rank(A)
print("Matrix:\n", A)
print("Rank of A:", rank)

# Eigenvalues and Eigenvectors
#Square matrix
M = np.array([[2, 1], [1, 2]])
eigvals, eigvecs = np.linalg.eig(M)
print("Eigenvalues:", eigvals)
print("Eigenvectors:", eigvecs)

# Visualize Eigenvectors

origin = [0], [0]
plt.quiver(*origin, eigvecs[0], eigvecs[1], color=['r', 'b'], scale=3)
plt.title("Eigenvectors of Matrix M")
plt.grid()
plt.axis('equal')
plt.show()
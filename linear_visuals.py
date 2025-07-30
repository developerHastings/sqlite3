import numpy as np
import matplotlib.pyplot as plt

M = np.array([[2, 1],
              [1, 2]])

eigvals, eigvecs = np.linalg.eig(M)

origin = np.array([[0, 0], [0, 0]])  # X and Y starting points for the arrows

# Extrat eigenvectors as columns
eig_vec1 = eigvecs[:, 0]
eig_vec2 = eigvecs[:, 1]

#Start point arrays (both start at 0,0)
X = [0, 0]
Y = [0, 0]

# U components: x-direction (first element of each eigenvector)
U = [eig_vec1[0], eig_vec2[0]]
# V components: y-direction (second element of each eigenvector)
V = [eig_vec1[1], eig_vec2[1]]

plt.quiver(X, Y, U, V, color=['r', 'b'], scale=3)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.title("Eigenvectors of Matrix M")
plt.grid()  
plt.show()
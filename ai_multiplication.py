import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
 
#First Method: np.dot
C = np.dot(A, B)
print("A dot B:\n", C)

# Second Method: @ operator (preffered in modern Python)
C2 = A @ B
print("A @ B:\n", C2)

# Creating Special Matrices
I = np.eye(2)  # Identity matrix of size 2x2
D = np.diag([2, 5])  # Diagonal matrix with specified diagonal elements
Z = np.zeros((2, 2))  # Zero matrix of size 2x2
print("Identity Matrix I:\n", I)
print("Diagonal Matrix D:\n", D)
print("Zero Matrix Z:\n", Z)

#Applying Transformations: 2D Example

#Points forming a square
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]). T  

#Scaling transformation: double size
S = np.array([[2, 0], [0, 2]])
scaled_square = S @ square

plt.plot(square[0], square[1], 'bo-', label='Original Square')
plt.plot(scaled_square[0], scaled_square[1], 'ro-', label='Scaled Square')
plt.legend()
plt.axis('equal')
plt.show()
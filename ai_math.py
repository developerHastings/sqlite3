import numpy as np

# 1D Vector
v = np.array([1,2,3])
print("Vector v:", v)

# 2D Matrix
M = np.array([[1, 2], [3, 4]])
print("Matrix M:\n", M)

#Basic addition Vector Addition
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print("v1 + v2", v1 + v2)

# Scalar Multiplication
print("3 * v1 =", 3 * v1)
print("2 * M =", 2 * M)

# Matrix addition
M2 = np.array([[5, 6], [7, 8]])
print("M + M2:\n", M + M2)

# A (Indexing and Slicing)
# Accessing specific elements
print("First element of v1:", v1[0])
print("Element at first row, second column of M:", M[0, 1])

# Access a range or subset
print("Slice of v1 (first two elements):", v1[0:2])
print("First row of M:", M[0, :])
print("Second column of M:", M[:, 1])



import numpy as np

# Create two vectors of size 4
vector1 = np.array([1, 2, 3, 4])
vector2 = np.array([5, 6, 7, 8])

# Add the vectors
add_vectors = vector1 + vector2

# Subtract the vectors
subtract_vectors = vector1 - vector2

# Multiply a vector by a scalar
scalar = 3
scalar_mult_vector = scalar * vector1

# Create two 3x3 matrices
matrix1 = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

matrix2 = np.array([[9, 8, 7],
                    [6, 5, 4],
                    [3, 2, 1]])

# Add the matrices
add_matrices = matrix1 + matrix2

# Multiply a matrix by a scalar
scalar_mult_matrix = scalar * matrix1

# Indexing
second_element_vector = vector1[1]  # Index starts at 0
element_first_row_third_col_matrix = matrix1[0, 2]

# Slicing
slice_vector = vector1[:2]  # first two elements
slice_matrix = matrix1[:2, :]  # first two rows, all columns

# Print results
print("Vector 1:", vector1)
print("Vector 2:", vector2)
print("Added vectors:", add_vectors)
print("Subtracted vectors:", subtract_vectors)
print("Scalar multiplied vector:", scalar_mult_vector)
print("\nMatrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)
print("Added matrices:\n", add_matrices)
print("Scalar multiplied matrix:\n", scalar_mult_matrix)
print("\nSecond element of vector1:", second_element_vector)
print("Element in first row, third column of matrix1:", element_first_row_third_col_matrix)
print("First two elements of vector1:", slice_vector)
print("First two rows of matrix1:\n", slice_matrix)

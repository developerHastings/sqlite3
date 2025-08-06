import numpy as np

# 1. Create two vectors (size 4)
v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])

# Vector addition and subtraction
v_add = v1 + v2
v_sub = v1 - v2

# Multiply a vector by a scalar
scalar = 3
v_scaled = scalar * v1

# 2. Create two 3×3 matrices
m1 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
m2 = np.array([[9, 8, 7],
               [6, 5, 4],
               [3, 2, 1]])

# Matrix addition
m_add = m1 + m2

# Multiply a matrix by a scalar
m_scaled = 2 * m2

# 3. Indexing examples:
second_element_v1 = v1[1]            # second element of vector v1
element_m1_1_3 = m1[0, 2]           # first row, third column (0‑based indexing)

# 4. Slicing examples:
v1_first_two = v1[:2]               # first two elements of vector
m1_first_two_rows = m1[:2, :]       # first two rows of matrix

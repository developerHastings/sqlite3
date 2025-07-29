import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Multiply Two 2x2 Matrices
# ---------------------------
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Multiply manually (for verification)
manual_product = np.array([[1*5 + 2*7, 1*6 + 2*8],
                           [3*5 + 4*7, 3*6 + 4*8]])

# Multiply using NumPy
numpy_product = np.dot(A, B)

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Manual A x B:\n", manual_product)
print("NumPy A x B:\n", numpy_product)

# ---------------------------
# 2. Identity and Zero Matrix
# ---------------------------
I = np.eye(2)
Z = np.zeros((2, 2))

print("\nIdentity Matrix I:\n", I)
print("Zero Matrix Z:\n", Z)
print("A x I:\n", np.dot(A, I))
print("A + Z:\n", A + Z)

# ---------------------------
# 3. 2D Transformations
# ---------------------------
# Original square shape (points in columns)
points = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])  # Shape: (2, 5)

# Scaling matrix
scale = np.array([[2, 0],
                  [0, 1]])

# Rotation matrix (30 degrees)
theta = np.radians(30)
rotate = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])

# Reflection over x-axis
reflect_x = np.array([[1, 0],
                      [0, -1]])

# Shear in x-direction
shear_x = np.array([[1, 1.5],
                    [0, 1]])

# Apply transformations
scaled_points = scale @ points
rotated_points = rotate @ points
reflected_points = reflect_x @ points
sheared_points = shear_x @ points

# ---------------------------
# 4. Plot the Transformations
# ---------------------------
plt.figure(figsize=(10, 8))

# Original
plt.plot(*points, 'k-', label='Original')

# Transformed shapes
plt.plot(*scaled_points, 'r--', label='Scaled (x2, y1)')
plt.plot(*rotated_points, 'g-.', label='Rotated (30°)')
plt.plot(*reflected_points, 'b:', label='Reflected (x-axis)')
plt.plot(*sheared_points, 'm-', label='Sheared (x-direction)')

# Display
plt.title("2D Transformations: Scaling, Rotation, Reflection, Shearing")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# ========== 1. CREATE MATRICES AND CHECK RANK ==========
print("---- Matrix Ranks ----")
A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 2], [2, 4]])  # Rank-deficient
C = np.array([[0, 1], [-1, 0]])  # Rotation matrix

matrices = {'A': A, 'B': B, 'C': C}
for name, M in matrices.items():
    rank = np.linalg.matrix_rank(M)
    print(f"Rank of {name}: {rank}")

# ========== 2. EIGENVALUES AND EIGENVECTORS ==========
print("\n---- Eigenvalues and Eigenvectors ----")

def eigen_info(M, label):
    eigvals, eigvecs = np.linalg.eig(M)
    print(f"\nMatrix {label}:\n{M}")
    print(f"Eigenvalues: {eigvals}")
    print(f"Eigenvectors:\n{eigvecs}")
    return eigvals, eigvecs

eigvals_A, eigvecs_A = eigen_info(A, 'A')

# ========== 3. VISUALIZE MATRIX TRANSFORMATION ==========
print("\n---- Visualization ----")

def plot_transformation(M, eigvecs, title):
    # Create a grid of vectors
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    # Plot unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.vstack((np.cos(theta), np.sin(theta)))
    transformed_circle = M @ circle

    ax.plot(circle[0], circle[1], 'b--', label='Original Circle')
    ax.plot(transformed_circle[0], transformed_circle[1], 'r-', label='Transformed Shape')

    # Plot eigenvectors
    origin = np.array([[0, 0], [0, 0]])  # origin point
    ax.quiver(*origin, eigvecs[0], eigvecs[1], color=['green', 'purple'], scale=1, scale_units='xy', angles='xy')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title(f'Transformation by Matrix: {title}')
    ax.legend()
    plt.grid(True)
    plt.show()

plot_transformation(A, eigvecs_A, 'A')


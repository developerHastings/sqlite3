import numpy as np
import matplotlib.pyplot as plt

def numerical_derivative(f, x, dx=1e-5):
    return (f(x + dx) - f(x - dx)) / (2 * dx)

def f(x):
    return 3 * x**3 + 2 * x

# Plot function and derivative
x_vals = np.linspace(-3, 3, 400)
y_vals = f(x_vals)
dy_vals = numerical_derivative(f, x_vals)

plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, label='f(x) = 3x³ + 2x')
plt.plot(x_vals, dy_vals, '--', label="Numerical f'(x)")
plt.title('Function and its Numerical Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Matrix transformation visualization
A = np.array([[2, 1],
              [1, 3]])

theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])  # unit circle points
ellipse = A @ circle

eigvals, eigvecs = np.linalg.eig(A)

plt.subplot(1, 2, 2)
plt.plot(circle[0], circle[1], label='Unit Circle')
plt.plot(ellipse[0], ellipse[1], label='Transformed Ellipse')
plt.quiver(0, 0, eigvecs[0,0], eigvecs[1,0], color='r', scale=3, label='Eigenvector 1')
plt.quiver(0, 0, eigvecs[0,1], eigvecs[1,1], color='g', scale=3, label='Eigenvector 2')
plt.axis('equal')
plt.title('Matrix Transformation and Eigenvectors')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# (b) Define the function f(x, y)
def f(x, y):
    return 3 * x**2 * y + 2 * y**3

# (c) Numerical partial derivatives using central difference
def partial_derivatives(f, x, y, h=1e-5):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return df_dx, df_dy

# Evaluate at (x=1, y=2)
x0, y0 = 1.0, 2.0
df_dx_num, df_dy_num = partial_derivatives(f, x0, y0)

# (a) Analytical derivatives at (1, 2)
df_dx_analytic = 6 * x0 * y0       # = 6 * 1 * 2 = 12
df_dy_analytic = 3 * x0**2 + 6 * y0**2  # = 3*1 + 6*4 = 3 + 24 = 27

# Print results
print("Analytical ∂f/∂x:", df_dx_analytic)
print("Analytical ∂f/∂y:", df_dy_analytic)
print("Numerical ∂f/∂x:", df_dx_num)
print("Numerical ∂f/∂y:", df_dy_num)

# Visualization
x = np.linspace(0, 2, 100)
y = np.linspace(0, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Surface plot of f(x, y)
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# Plot the point (x0, y0, f(x0, y0))
z0 = f(x0, y0)
ax.scatter(x0, y0, z0, color='r', s=50, label='Point (1,2)')

# Plot gradient vector at (x0, y0)
# We scale the vector for better visualization
scale = 0.1
ax.quiver(x0, y0, z0,
          df_dx_num, df_dy_num, 0,
          length=scale, color='r', label='Gradient vector')

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Function f(x, y) and Gradient at (1, 2)')
ax.legend()

plt.show()

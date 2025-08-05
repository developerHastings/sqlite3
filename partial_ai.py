import numpy as np

# f(x, y) = x^2 + y^2
def f_xy(x, y):
    return x**2 + y**2

# Analytical partials
# df/dx = 2x, df/dy = 2y

# Evaluate at (x, y) = (1.0, 2.0)
x0, y0 = 1.0, 2.0
df_dx = 2 * x0
df_dy = 2 * y0
print("Partial df/dx at (1,2):", df_dx)
print("Partial df/dy at (1,2):", df_dy)

#Numerical gradient with NumPy: finite differences
def numerical_gradient(f, x, y, h=1e-5):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

grad = numerical_gradient(f_xy, x0, y0)
print("Numerical gradient at (1,2):", grad)
import numpy as np

def numerical_derivative(f, x, dx=1e-5):
    return (f(x + dx) - f(x - dx)) / (2 * dx)

def f(x):
    return 3 * x**3 + 2 * x

points = [-2, -1, 0, 1, 2, 3]

for x in points:
    deriv = numerical_derivative(f, x)
    print(f"f'({x}) ≈ {deriv:.6f}")

import numpy as np

def numerical_derivative(f, x, dx=1e-5):
    return (f(x + dx) - f(x - dx)) / (2 * dx)

def my_function(x):
    return 3 * x**3 + 2 * x

x_point = 1
result = numerical_derivative(my_function, x_point)

print(f"The derivative of f(x) at x = {x_point} is approximately {result:.6f}")

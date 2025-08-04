import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

x = np.linspace(-3, 3, 100)
y = f(x)

# Numerical derivative using finite differences
dx = 1e-4
dy_dx = (f(x + dx) - f(x)) / dx

# Plotting the function and its derivative
plt.plot(x, y, label='y = x^2')
plt.plot(x, dy_dx, label='Numerical Derivative')
plt.legend()
plt.title('Function and its Numerical Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()
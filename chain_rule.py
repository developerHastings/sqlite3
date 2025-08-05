import numpy as np
import matplotlib.pyplot as plt

# f(x) = sin(x^2)
def f(x):
    return np.sin(x**2)

# Dreivative using the chain rule: f'(x) = 2x * cos(x^2)
def f_prime(x):
    return 2 * x * np.cos(x**2)

x = np.linspace(-2, 2, 100)

plt.plot(x, f(x), label='f(x) = sin(x^2)')
plt.plot(x, f_prime(x), label="f'(x)", linestyle='--')
plt.legend()
plt.show()


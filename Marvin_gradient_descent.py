import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return x**2 + 2*x + 1

def df(x):
    return 2*x + 2  # derivative of f(x)

# Gradient Descent parameters
x = 5.0             # Initial guess
learning_rate = 0.1
iterations = 20

# Lists to store the descent path
x_values = [x]
f_values = [f(x)]

# Perform Gradient Descent
for i in range(iterations):
    grad = df(x)
    x = x - learning_rate * grad
    x_values.append(x)
    f_values.append(f(x))

# Plot the function
x_plot = np.linspace(-5, 5, 400)
y_plot = f(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='f(x) = x² + 2x + 1')
plt.plot(x_values, f_values, 'ro-', label='Gradient Descent Path')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Objective Value Across Gradient Descent')  # <- Updated Title
plt.legend()
plt.grid(True)
plt.show()

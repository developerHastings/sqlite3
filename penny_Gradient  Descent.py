import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return x**2 + 2*x + 1

def df(x):
    return 2*x + 2

# Gradient descent parameters
x = 3.0                # Initial guess
learning_rate = 0.1
num_iterations = 25

# Store the path for plotting
x_vals = [x]
f_vals = [f(x)]

# Gradient descent loop
for _ in range(num_iterations):
    x = x - learning_rate * df(x)
    x_vals.append(x)
    f_vals.append(f(x))

# Plotting the function
x_plot = np.linspace(-4, 4, 200)
y_plot = f(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot, label='f(x) = x² + 2x + 1', color='blue')
plt.scatter(x_vals, f_vals, color='red', label='Gradient Descent Steps')

# Draw arrows between steps
for i in range(len(x_vals) - 1):
    plt.annotate('', 
                 xy=(x_vals[i+1], f_vals[i+1]), 
                 xytext=(x_vals[i], f_vals[i]),
                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent on f(x) = x² + 2x + 1')
plt.legend()
plt.grid(True)
plt.show()

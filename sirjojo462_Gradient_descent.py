import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return x**2 + 2*x + 1

def df(x):
    return 2*x + 2

# Combined implementation & visualization
def gradient_descent_plot(start_x, learning_rate, iterations, tolerance=None):
    x = start_x
    history = [x]
    for i in range(iterations):
        grad = df(x)
        x_new = x - learning_rate * grad
        history.append(x_new)
        if tolerance is not None and abs(x_new - x) < tolerance:
            break
        x = x_new

    # Plot the function curve
    xs = np.linspace(start_x - 5, start_x + 5, 400)
    ys = f(xs)
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, label='f(x) = x² + 2x + 1', color='blue')

    # Plot the descent path
    history = np.array(history)
    plt.scatter(history, f(history), color='red', marker='o', label='Descent steps')
    plt.plot(history, f(history), color='red', linestyle='--', alpha=0.5)

    # Draw a vertical line at the true minimum
    plt.axvline(-1, color='green', linestyle=':', label='True minimum x = -1')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.title(f'Gradient Descent Path (start x={start_x}, η={learning_rate})')
    plt.show()

    return history, f(history[-1])

# Example run:
history, f_min = gradient_descent_plot(start_x=5.0, learning_rate=0.1,
                                      iterations=50, tolerance=1e-6)
print(f"Converged to x = {history[-1]:.6f}, f(x) = {f_min:.6f}")

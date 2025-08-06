import numpy as np
import matplotlib.pyplot as plt 

# Define the function and its derivative
def f(x):
    return (x - 3)**2 + 4 #Minimum at x=3

def df_dx(x):
    return 2 * (x - 3) #Derivative of f(x)


# Implementing gradient descent
x = 0.0 # Starting point
alpha = 0.1 # Learning rate
steps = 20
vec_x = [x] # to log all positions

for step in range(steps):
    grad = df_dx(x) # Compute the gradient
    x = x - alpha * grad
    vec_x.append(x) # Log the position
    print(f"Step {step+1}: x = {x:.4f}, f(x) = {f(x):.4f}")


    # Plot the trajectory of x over steps
    plt.figure(figsize=(8,5))
    plt.plot([f(xx) for xx in vec_x], marker='o')
    plt.title('Objective Value Acrosss Gradient Descent Steps')
    plt.xlabel('Step')
    plt.ylabel('f(x)')
    plt.show()

    


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data: x (feature), y (target)
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

# Manual calculation of regression coefficients
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate the slope (beta_1)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
beta_1 = numerator / denominator

# Calculate the intercept (beta_0)
beta_0 = y_mean - beta_1 * x_mean

print(f"Manual calculation: intercept = {beta_0:.2f}, slope = {beta_1:.2f}")

# Predicted values
y_pred_manual = beta_0 + beta_1 * x

# using scikit-learn for comparison
x_reshaped = x.reshape(-1, 1) # sklearn expects 2D input
model = LinearRegression().fit(x_reshaped, y)
print(f"Sklearn calculation: intercept = {model.intercept_:.2f}, slope = {model.coef_[0]:.2f}")

y_pred_sklearn = model.predict(x_reshaped)

# Plotting actual data and fitted lines
plt.scatter(x, y, color='blue', label='Actual data')
plt.plot(x, y_pred_manual, color='red', label='Manual Fit')
plt.plot(x, y_pred_sklearn, color='green', linestyle='--', label='Sklearn Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Linear Regression: Manual vs Sklearn')
plt.legend()
plt.show()
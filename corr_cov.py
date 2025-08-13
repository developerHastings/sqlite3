import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data: 100 samples for two variables with some relation
np.random.seed(0)
x = np.random.normal(0, 1, 100)
y = 2.0 * x + np.random.normal(0, 1, 100)  # y roughly correlates with x

# Calculate covariance matrix
cov_matrix = np.cov(x, y)
print("Covariance matrix:\n", cov_matrix)

# Calculate correlation matrix
corr_matrix = np.corrcoef(x, y)
print("Correlation matrix:\n", corr_matrix)

# Scatter plot to visualize the relationship
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of x vs y')
plt.show()



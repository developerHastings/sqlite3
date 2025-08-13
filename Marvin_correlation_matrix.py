import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)



# Low correlation: x random, y random + small relationship
x_low = np.random.normal(0, 1, 100)
y_low = np.random.normal(0, 1, 100) * 0.5 + np.random.normal(0, 1, 100)

# High correlation: y = x with small noise
x_high = np.random.normal(0, 1, 100)
y_high = x_high * 2 + np.random.normal(0, 0.2, 100)



low_df = pd.DataFrame({'X_low': x_low, 'Y_low': y_low})
high_df = pd.DataFrame({'X_high': x_high, 'Y_high': y_high})

cov_low = low_df.cov()
corr_low = low_df.corr()

cov_high = high_df.cov()
corr_high = high_df.corr()

# Print results
print("Low correlation - Covariance Matrix:\n", cov_low, "\n")
print("Low correlation - Correlation Matrix:\n", corr_low, "\n")
print("High correlation - Covariance Matrix:\n", cov_high, "\n")
print("High correlation - Correlation Matrix:\n", corr_high, "\n")


plt.figure(figsize=(10,4))

# Low correlation scatter
plt.subplot(1, 2, 1)
plt.scatter(x_low, y_low, alpha=0.7, color='blue')
plt.title("Low Correlation")
plt.xlabel("X_low")
plt.ylabel("Y_low")

# High correlation scatter
plt.subplot(1, 2, 2)
plt.scatter(x_high, y_high, alpha=0.7, color='red')
plt.title("High Correlation")
plt.xlabel("X_high")
plt.ylabel("Y_high")

plt.tight_layout()
plt.show()

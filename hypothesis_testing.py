import numpy as np
from scipy import stats

# Simulate sample data: plant heights (in cm)
np.random.seed(1)
sample_data = np.random.normal(loc=52, scale=5, size=30) # Mean= 51, std= 5, n= 30

# Define the population mean under the null hypothesis
popmean = 50

# Perform one-sample t-test
t_stat, p_value = stats.ttest_1samp(sample_data, popmean)

# Print the results
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The sample mean is significantly different from the population mean or 50 cm.")
else:
    print("Fail to reject the null hypothesis: no significant difference detected.")
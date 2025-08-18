import numpy as np

# Suppose population is 10,000 values, normally distributed
population = np.random.normal(loc=50, scale=10, size=10000)

# Draw a random sample of 100
sample = np.random.choice(population, size=100, replace=False)

# Estimate mean and standard deviation
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
print("Sample mean:", sample_mean)

# 95% confidence interval for the sample mean
n = len(sample)
ci_lower = sample_mean - 1.96 * (sample_std / np.sqrt(n))
ci_upper = sample_mean + 1.96 * (sample_std / np.sqrt(n))
print("95% Confidence interval: [{:.2f}, {:.2f}]".format(ci_lower, ci_upper))
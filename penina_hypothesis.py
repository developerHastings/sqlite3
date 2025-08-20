import numpy as np
from scipy import stats

true_mean = 65           # Mean of the simulated population
population_mean = 60     # Population mean to test against (null hypothesis)
std_dev = 10             # Standard deviation of the data
sample_sizes = [10, 30, 50, 100, 500, 1000]  # Sample sizes to test
alpha = 0.05             # Significance level

# For reproducibility
np.random.seed(42)



# Header
print(f"{'Sample Size':>12} | {'Sample Mean':>12} | {'T-Statistic':>11} | {'P-Value':>8} | {'Significant':>11}")
print("-" * 65)

# Run simulation and test for each sample size
for size in sample_sizes:
    # Generate sample data
    sample = np.random.normal(loc=true_mean, scale=std_dev, size=size)

    # Perform one-sample t-test
    t_stat, p_value = stats.ttest_1samp(sample, population_mean)
    sample_mean = np.mean(sample)

    # Determine significance
    significant = p_value < alpha

    # Print results
    print(f"{size:12} | {sample_mean:12.2f} | {t_stat:11.4f} | {p_value:8.4f} | {str(significant):>11}")

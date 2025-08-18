import numpy as np

# Simulate a population
np.random.seed(42)
population_mean = 50
population_std = 12
population_size = 100000

population = np.random.normal(loc=population_mean, scale=population_std, size=population_size)

# Function to calculate 95% confidence interval using only numpy
def confidence_interval(sample, confidence=0.95):
    n = len(sample)
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)  # sample standard deviation
    z = 1.96  # 95% confidence, normal approximation
    margin = z * (std / np.sqrt(n))
    return mean, (mean - margin, mean + margin)

# Draw samples of different sizes
sample_sizes = [10, 100, 1000]

for size in sample_sizes:
    sample = np.random.choice(population, size=size, replace=False)
    mean, ci = confidence_interval(sample)
    print(f"Sample size: {size}")
    print(f"Sample mean: {mean:.2f}")
    print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
    print("-" * 40)

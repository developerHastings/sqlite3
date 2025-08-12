import numpy as np
import matplotlib.pyplot as plt

# Simulate tossing two coins 10000 times
# Each toss can be 0 (tails) or 1 (heads)
tosses = np.random.randint(0, 2, (10000, 2))
# Sum heads in each toss pair
heads_count = np.sum(tosses, axis=1)

# Calculate probabilities for 0, 1, and 2 heads
values, counts = np.unique(heads_count, return_counts=True)
probabilities = counts / counts.sum()

print("Possible heads count:", values)
print("Estimated probabilities:", probabilities)

# Plotting probablity distribution as bar chart
plt.bar(values, probabilities, tick_label=values)
plt.xlabel('Number of Heads')
plt.ylabel('Probability')
plt.title('Probability Distribution: Number of Heads in Two Coin Tosses')
plt.show()


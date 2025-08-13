import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_trials = 10_000
rng = np.random.default_rng()

# Simulate rolling two dice
die1 = rng.integers(1, 7, size=n_trials)
die2 = rng.integers(1, 7, size=n_trials)
sums = die1 + die2

# Compute probability distribution
possible_sums = np.arange(2, 13)
counts = np.array([np.sum(sums == s) for s in possible_sums])
probabilities = counts / n_trials

# Print results table
print(f"Total trials: {n_trials}")
print("Sum  Count  Probability")
for s, c, p in zip(possible_sums, counts, probabilities):
    print(f"{s:>3}  {c:>5}  {p:.4f}")

# Identify and print most likely sums
max_prob = probabilities.max()
most_likely_sums = possible_sums[probabilities == max_prob]
print(f"\nMost likely sum(s): {list(most_likely_sums)} with probability {max_prob:.4f}")
print("Theoretical most likely sum: 7 (probability ≈ 0.1667)")

# Plot probability distribution
plt.figure(figsize=(8, 4.5))
plt.bar(possible_sums, probabilities)
plt.xticks(possible_sums)
plt.xlabel("Sum of two dice")
plt.ylabel("Estimated probability")
plt.title("Probability distribution of sum (two dice)")
plt.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.show()

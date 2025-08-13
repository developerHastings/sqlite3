import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def simulate_rolls(n_rolls):
    """Simulate rolling two fair dice n_rolls times."""
    return [random.randint(1, 6) + random.randint(1, 6) for _ in range(n_rolls)]

def compute_probabilities(rolls):
    """Compute probability distribution from list of sums."""
    counts = Counter(rolls)
    total = len(rolls)
    # sums from 2 to 12 inclusive
    return {s: counts[s] / total for s in range(2, 13)}

def plot_distribution(probabilities):
    """Plot a bar chart of the probability distribution."""
    sums = list(sorted(probabilities.keys()))
    probs = [probabilities[s] for s in sums]
    
    plt.figure(figsize=(8, 5))
    plt.bar(sums, probs, align='center', color='skyblue')
    plt.xlabel('Sum of Two Dice')
    plt.ylabel('Probability')
    plt.title(f'Probability Distribution of Two-Dice Sums (Simulation: {sum(probabilities.values())*len(sums):.0f} rolls)')
    plt.xticks(sums)
    plt.ylim(0, max(probs) * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def find_most_likely(probabilities):
    """Return the sum(s) with the highest probability."""
    max_prob = max(probabilities.values())
    most_likely = [s for s, p in probabilities.items() if p == max_prob]
    return most_likely, max_prob

def main():
    n_rolls = 10_000
    rolls = simulate_rolls(n_rolls)
    probabilities = compute_probabilities(rolls)
    plot_distribution(probabilities)
    
    most_likely, max_p = find_most_likely(probabilities)
    print(f"Most likely sum(s): {most_likely} with estimated probability {max_p:.4f}")

if __name__ == '__main__':
    main()

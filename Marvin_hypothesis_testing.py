import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --- PARAMETERS ---
np.random.seed(42)     # for reproducibility
mu0 = 50.0             # null hypothesis mean
true_mean = 52.0       # true mean (alternative)
sigma = 10.0           # true std dev for simulation
n0 = 40                # initial sample size
alpha = 0.05           # significance level
R = 1000               # repetitions for power estimation

# --- 1) Single-sample test ---
sample = np.random.normal(loc=true_mean, scale=sigma, size=n0)

xbar = np.mean(sample)
s = np.std(sample, ddof=1)
t_stat, p_val = stats.ttest_1samp(sample, popmean=mu0)

print("Single-sample test (n = {}):".format(n0))
print(f"  Null mean (mu0): {mu0:.2f}")
print(f"  Sample mean:     {xbar:.2f}")
print(f"  Sample SD:       {s:.2f}")
print(f"  t-statistic:     {t_stat:.4f}")
print(f"  p-value (two-sided): {p_val:.6f}")
print(f"  Significant at alpha=0.05? {'YES' if p_val < 0.05 else 'NO'}\n")

# --- 2) Vary sample size and estimate power ---
n_list = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
rows = []

for n in n_list:
    # Single run
    sample_n = np.random.normal(loc=true_mean, scale=sigma, size=n)
    t_single, p_single = stats.ttest_1samp(sample_n, popmean=mu0)
    sig_single = p_single < alpha

    # Power estimation
    rejections = 0
    for _ in range(R):
        sim = np.random.normal(loc=true_mean, scale=sigma, size=n)
        _, p_rep = stats.ttest_1samp(sim, popmean=mu0)
        if p_rep < alpha:
            rejections += 1
    power = rejections / R

    rows.append({
        "n": n,
        "p_value_single_run": p_single,
        "significant_single_run": sig_single,
        "power_estimate": power
    })

results_df = pd.DataFrame(rows)
print(results_df)

# --- 3) Plot estimated power vs sample size ---
plt.figure()
plt.plot(results_df["n"], results_df["power_estimate"], marker="o")
plt.title("Estimated Power vs Sample Size (alpha = 0.05)\nTrue mean = 52, Null mean = 50, SD = 10")
plt.xlabel("Sample size (n)")
plt.ylabel("Estimated power (rejection rate)")
plt.grid(True)
plt.show()

# --- 4) Plot single-run p-value vs sample size ---
plt.figure()
plt.plot(results_df["n"], results_df["p_value_single_run"], marker="o")
plt.axhline(alpha, linestyle="--", color="red")
plt.title("Single-run p-value vs Sample Size (alpha = 0.05)")
plt.xlabel("Sample size (n)")
plt.ylabel("Single-run p-value")
plt.grid(True)
plt.show()

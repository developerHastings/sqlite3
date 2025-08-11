import numpy as np

# Simulate 1000 coin flips: 0 = tails, 1 = heads
flips = np.random.randint(0, 2, size=1000)

# Estimate probability of heads
prob_heads = np.mean(flips)
print("Estimated probability of heads:", prob_heads)

# Simulate rolling a dice 1000 times
dice_rolls = np.random.randint(1, 7, size=1000)

# Estimate probability of rolling a 6
prob_six = np.mean(dice_rolls == 6)
print("Estimated probability of rolling a 6:", prob_six)


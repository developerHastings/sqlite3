import numpy as np  

#Create data
x = np.random.rand(100, 2)
y = np.random.randint(0, 2, size=100)

# Normalize features
X_min = x.min(axis=0)
X_max = x.max(axis=0)
X_norm = (x - X_min) / (X_max - X_min)

# Split train/test (80/20)

X_train = X_norm[:80]
X_test = X_norm[80:]
y_train = y[:80]
y_test = y[80:]

# Basic statistics
print("Train mean:", X_train.mean(axis=0))
print("Train max:", X_train.max(axis=0))

#Add bias column (ones) to features
bias = np.ones((X_train.shape[0], 1))
X_train_bias = np.hstack([X_train, bias])


print("Train features with bias:\n", X_train_bias[:3])
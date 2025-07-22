import pandas as pd

# 1. Load data
df = pd.read_csv("iris.csv")

# 2. Basic inspection
print("First 5 rows of data:")
print(df.head())

print("\nStatistical summary:")
print(df.describe())

# 3. Handle missing data
df = df.dropna()

# 4. Encode string labels to numbers
df["species_code"] = df["species"].astype("category").cat.codes

# 5. Select features and target
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species_code"]

print("\nSelected Features (first 5 rows):")
print(X.head())
print("\nEncoded Target (first 5 values):")
print(y.head())

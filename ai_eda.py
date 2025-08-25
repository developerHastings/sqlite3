import pandas as pd

# Load dataset
df = pd.read_csv('mess.csv')

# Inspect dataset
print(df.info())
print(df.describe(include='all'))
print(df.head())

# Find missing values
print(df.isnull().sum())

# Preview duplicated rows
print(df[df.duplicated()])

# Check unique city names (to detect inconsistent entries))
print(df['city'].unique())

# Check 'age' data for anomalies
print(df['age'].unique())


import pandas as pd

# URL of Iris dataset from UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Define column names as per the dataset description
col_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Load dataset directly into a DataFrame
iris = pd.read_csv(url, header=None, names=col_names)

# Display the first 5 rows
print(iris.head())

# Optionally save to local CSV file:
iris.to_csv("iris.csv", index=False)
print("Iris dataset saved locally as iris.csv")

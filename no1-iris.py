from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
X = iris.data       
y = iris.target     
feature_names = iris.feature_names

# Step 3: Display basic information

# Shape of the dataset
print("Shape of X (features):", X.shape)
print("Shape of y (target):", y.shape)


print("Feature names:", feature_names)

# Convert to DataFrame for easier viewing
iris_df = pd.DataFrame(X, columns=feature_names)
iris_df['species'] = [iris.target_names[i] for i in y]

# Show first 5 rows
print("\nFirst 5 rows of the dataset:")
print(iris_df.head())

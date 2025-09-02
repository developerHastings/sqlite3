import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data: house size (sq ft) vs price (1000s $)
data = {
    'size_sqft': [750, 800, 850, 900, 950, 1000, 1050, 1100],
    'price_k': [150, 160, 165, 170, 180, 190, 200, 210]
}

# Create a DatFrame from the data
df = pd.DataFrame(data)

# Features (input variables) and labels (output variable)
X = df[['size_sqft']] # Feature: house size
y = df['price_k']   # Label: house price

# Split the data into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict house prices using the test set features
y_pred = model.predict(X_test)

# Plot the training data, test data, and model predictions
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X_test, y_pred, color='red', label='Model predictions')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price (1000s $)')
plt.title('House Price Prediction using Linear Regression')
plt.legend()
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Sample fruit classification data
data = {
    'size': [3, 5, 7, 2, 6, 7, 4], # Feature size of the fruit (arbitary units)
    'color': ['red', 'orange', 'orange', 'red', 'yellow', 'yellow', 'red'], # Feature: color of fruit
    'label': ['apple', 'orange', 'orange', 'apple', 'banana', 'banana', 'apple'] # Labels: type of fruit (target)
}

# Create a DataFrame (table) from the data dictionary
df = pd.DataFrame(data)

# Convert the categorical 'color' feature into numbers using one-hot enconding
df_encoded = pd.get_dummies(df, columns=['color'])

# Separate features (inputs) and labels (target)
X = df_encoded.drop('label', axis=1) # Drop label column; remaining are features
y = df_encoded['label'] # Labels to predict

# Split the data into training and testing sets
# test_size=0.25 means 25% of data for testing; random_state=42 fixes randomness for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a Decision Tree Classifier Object
model = DecisionTreeClassifier(max_depth=3) # Limit tree depth to avoid overfitting

# Train the model using the training data
model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Calculate accuracy: percentage of correct predictions
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix: counts of true positives, false positives
conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Print accuracy and confusion matrix
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(pd.DataFrame(conf_matrix, index=model.classes_, columns=model.classes_))

# Visualize the decision tree
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.title('Decision Tree for Fruit Classification')
plt.show()
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Binary classification: Iris setosa vs. other

# Split train anfd test sets (70-30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize logistic regression model with max 200 iterations for convergence
logreg = LogisticRegression(max_iter=200)

# Train the model on training data
logreg.fit(X_train, y_train)

# Evaluate class labels for test samples
y_pred = logreg.predict(X_test)

# Evaluate classification performance
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

# Predict class probablities for positive class
probs = logreg.predict_proba(X_test)[:, 1]

# Plot histogram of predicted probabilities
plt.hist(probs, bins=10, edgecolor='k')
plt.title('Predicted Probabilities for Positive Class (Iris Setosa)')
plt.xlabel('Probability')
plt.ylabel('Number of Samples')
plt.show()
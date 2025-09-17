from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bagging: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}')

# Boosting: AdaBoost Classifier
adb = AdaBoostClassifier(n_estimators=100, random_state=42)
adb.fit(X_train, y_train)
y_pred_adb = adb.predict(X_test)
print("AdaBoost Classification Report:")
print(classification_report(y_test, y_pred_adb))
print(f'AdaBoost Accuracy: {accuracy_score(y_test, y_pred_adb):.2f}')
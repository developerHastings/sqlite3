import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# For simplicity, convert to a binary classification (class 0 vs. not)
y_binary = (y == 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Train uncalibrated decision tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict probabilities
prob_pos_uncalibrated = clf.predict_proba(X_test)[:, 1]

# Calibration curve without calibration
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos_uncalibrated, n_bins=10)

# Calibrate the classifier using isotonic regression
calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
calibrated_clf.fit(X_train, y_train)
prob_pos_calibrated = calibrated_clf.predict_proba(X_test)[:, 1]

# Calibration curve with calibration
fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(y_test, prob_pos_calibrated, n_bins=10)

# Brier scores before and after calibration
print("Brier score (uncalibrated): {:.3f}".format(brier_score_loss(y_test, prob_pos_uncalibrated)))
print("Brier score (calibrated): {:.3f}".format(brier_score_loss(y_test, prob_pos_calibrated)))

# Plot calibration curves
plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Uncalibrated")
plt.plot(mean_predicted_value_cal, fraction_of_positives_cal, "s-", label="Calibrated (Isotonic)")
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curves")
plt.legend()
plt.show()
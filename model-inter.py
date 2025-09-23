# SHAP (SHapley Additive exPlanations):
# SHAP Value Calculation:
# SHAP_value(feature_i) = Average contribution of feature_i across all possible feature combinations.

# LIME (Local Interpretable Model-agnostic Explanations):

# PDPs:
# PDP_feature_x(x) = Average prediction when feature_x = x, marginalizing over other features.

# =====================================================================================
# Model Interpretability Demo: SHAP and LIME
# =====================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=== Model Interpretability Demo: SHAP and LIME ===\n")

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
iris = load_iris()
X = iris.data # Feature matrix: 150 samples, 4 features
y = iris.target # Target labels: 0, 1, 2 (representing flower species)

# Display dataset information
print(f"Dataset shape: {X.shape}")
print(f"Feature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")
print(f"Number of classes: {len(iris.target_names)}")

# Create DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['species'] = [iris.target_names[i] for i in y]

print("\nFirst 5 samples:")
print(df.head())
print(f"\nTarget distribution:\n{df['target'].value_counts().sort_index()}")

# Split Data into Train/Test
print("\nStep 2: Split Data into Train/Test...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,   #30% test size
    random_state=42, #Seed for reproducibility
    stratify=y      #Maintain class distribution in splits
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training target distribution: {np.bincount(y_train)}")
print(f"Testing target distribution: {np.bincount(y_test)}")

# Train Random Forest Classifier
print("\nStep 3: Train Random Forest Classifier...")

# Initialize Random Forest model
rf = RandomForestClassifier(
    n_estimators=100, #Number of decision trees in the forest
    max_depth=3,    #Maximum depth of each tree
    random_state=42,   #Seed for reproducibility
    oob_score=True  #Calculate out-of-bag score for validation
)

# Train the model on training data
rf.fit(X_train, y_train)

# Make predictions on test data
y_pred = rf.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
oob_score = rf.oob_score_

print(f"Model training completed!")
print(f"Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Out-of-Bag Score: {oob_score:.3f}")

# Display some predictions vs actual values
print("\nSample predictions (first 10 test samples):")
print("Actual: ", y_test[:10])
print("Predicted:", y_pred[:10])
print("Match: ", y_test[:10] == y_pred[:10])

# SHAP ANALYSIS - Global and Local Interpretability
print("\n" + "="*60)
print("Step 4: SHAP Analysis")
print("="*60)

# Create SHAP Tree Explainer
print("Creating SHAP Tree Explainer...")
explainer = shap.TreeExplainer(rf)

# Calculate SHAP values for all test samples
print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_test)

print(f"SHAP Values computed successfully!")
print(f"SHAP Values shape: {np.array(shap_values).shape}") # (3 classes, 45 samples, 4 features)
print(f"Expected values (base values): {explainer.expected_value}")

# Global Interpretability: Summary Plot
print("\nGenerating SHAP Summary Plot (Global Interpretability)...")

plt.figure(figsize=(10,8))
shap.summary_plot(shap_values, X_test, feature_names=iris.feature_names, show=False)
plt.title("SHAP Summary Plot - Global Feature Importance", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# SHAP Force Plots - Local Interpretability for each class
print("\nGenerating SHAP Force Plots for Local Interpretability...")

# Analyze first test sample for each class
sample_idx = 0 # We'll examine the first test sample

print(f"\nAnalyzing sample index: {sample_idx}:")
print(f"Actual feature values:")

for i, name in enumerate(iris.feature_names):
    print(f"  {name}: {X_test[sample_idx, i]:.2f}")
print(f"True class: {iris.target_names[y_test[sample_idx]]}")
print(f"Predicted class: {iris.target_names[y_pred[sample_idx]]}")

# Generate force plots for each class
for class_idx, class_name in enumerate(iris.target_names):
    print(f"\n--- SHAP Force Plot for {class_name} class ---")

    plt.figure(figsize=(12,4))
    shap.force_plot(
        explainer.expected_value[class_idx], #Base value for the class
        shap_values[class_idx][sample_idx, :], #SHAP values for the sample
        X_test[sample_idx, :], #Feature values for the sample
        feature_names=iris.feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Force  Plot - {class_name} Class\n(Sample {sample_idx})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# LIME ANALYSIS - Local Interpretability
print("\n" + "="*60)
print("Step 5: LIME Analysis")
print("="*60)

# Create LIME Tabular Explainer
print("Creating LIME Tabular Explainer...")
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train,    #Training data for stats
    feature_names=iris.feature_names, #Feature names
    class_names=iris.target_names,  #Class names
    discretize_continuous=True, #Discretize continuous features
    random_state=42 #Seed for reproducibility
)

print("LIME Explainer created successfully!")

# Lime Explanation for Individual Prediction
print(f"\Generating LIME Explanation for sample index {sample_idx}...")

# Generate explanation for first test sample
exp = explainer_lime.explain_instance(
    X_test[sample_idx], #Instance to explain
    rf.predict_proba,  #Prediction function (returns probabilities)
    num_features=4,    #Number of features to include in explanation
    top_labels=3       #Number of top classes to explain
)

print("LIME Explanation generated successfully!")

# Display Lime Explanation in Notebook Format
print("\nLIME Explanation Display:")
print("="*40)

try:
    # Try to display in notebook format
    exp.show_in_notebook(show_table=False)
except:
    print("Notebook display not available - printing text version instead:")
    print(exp.as_list())


# Detailed Text-Based LIME Explanation
print("\nDetailed LIME Explanation:")
print("="*40)

# Get explanation as a mapping
exp_map = exp.as_map()

print(f"\nSample {sample_idx} details:")
print(f"True class: {iris.target_names[y_test[sample_idx]]}")
print(f"Predicted class: {iris.target_names[y_pred[sample_idx]]}")

# Print detailed feature contributions for each class
for class_idx, class_name in enumerate(iris.target_names):
    print(f"\n--- Feature contributions for {class_name} class ---")

    # Get features used in explanation for this class
    features_used = exp_map[class_idx]

    # Sort by absolute weight (most important first)
    sorted_features = sorted(features_used, key=lambda x: abs(x[1]), reverse=True)

    total_contribution = 0
    for feature_idx, weight in sorted_features:
        feature_name = iris.feature_names[feature_idx]
        actual_value = X_test[sample_idx, feature_idx]
        direction = "INCREASES" if weight > 0 else "DECREASES"
        print(f" {feature_name} = {actual_value:.2f}: {weight:+.4f} ({direction})")
        total_contribution += weight
    print(f" Total contribution: {total_contribution:+.4f}")

    # Comparison and Summary
    print("\n" + "="*60)
    print("Step 6: Comparison and Summary")
    print("="*60)

    # Calculate feature importance from SHAP
    shap_importance = np.abs(shap_values).mean(axis=1).mean(axis=0) # Mean absolute SHAP value
    print("\nFeature Importance Comparison:")
    print("-" * 40)

    print("SHAP-based importance (mean |SHAP value|):")
    for i, name in enumerate(iris.feature_names):
        print(f" {name}: {shap_importance[i]:.4f}")

    # Get LIME importance from our explanation
print("\nLIME-based importance (from sample 0):")
lime_weights = {}
for class_idx in exp_map:
    for feature_idx, weight in exp_map[class_idx]:
        feature_name = iris.feature_names[feature_idx]
        if feature_name not in lime_weights:
            lime_weights[feature_name] = 0
        lime_weights[feature_name] += abs(weight)

for feature_name in iris.feature_names:
    importance = lime_weights.get(feature_name, 0)
    print(f" {feature_name}: {importance:.4f}")


# Final Prediction Probabilities
print("\nFinal predicition Probabilities for sample 0:")
print("-" * 45)

probabilities = rf.predict_proba(X_test[sample_idx:sample_idx+1])[0]
for class_idx, class_name in enumerate(iris.target_names):
    print(f" {class_name}: {probabilities[class_idx]:.4f} ({probabilities[class_idx]*100:.1f}%)")

print(f"\nPredicted class: {iris.target_names[np.argmax(probabilities)]}")
print(f"Actual class: {iris.target_names[y_test[sample_idx]]}")

print("\n=== Model Interpretability Demo Completed Successfuly! ===")
"""
=============================================================================
NAVTTC Artificial Intelligence (Machine Learning & Deep Learning) Tasks Sheet
Tasks 51 to 56
File: Task 51-56.py
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.datasets import load_iris, load_breast_cancer

# =============================================================================
# TASK 51
# =============================================================================
"""
TASK 51: Machine Learning (Implement Linear Regression)
Reference Link: https://stackabuse.com/linear-regression-in-python-with-scikit-learn/
Statement:
Implement Linear Regression in Python using scikit-learn:
1. Prepare continuous numerical dataset (features & target)
2. Train-test split
3. Train LinearRegression model
4. Inspect model coefficients and intercept
5. Make predictions and evaluate using MAE, MSE, RMSE, and R2 score
Week: 5
"""
print("=" * 70)
print("TASK 51: Machine Learning - Linear Regression Implementation")
print("=" * 70)

# 1. Dataset: Student Study Hours & Attendance predicting Exam Score
np.random.seed(42)
n_samples = 100
study_hours = np.random.uniform(1.0, 10.0, n_samples)
attendance_rate = np.random.uniform(60.0, 100.0, n_samples)
# True relationship: score = 15 + 5.5 * study_hours + 0.3 * attendance + noise
exam_score = 15 + (5.5 * study_hours) + (0.3 * attendance_rate) + np.random.normal(0, 2.5, n_samples)

X_lr = np.column_stack([study_hours, attendance_rate])
y_lr = exam_score

# 2. Train-test split
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

# 3. Model training
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)

# 4. Coefficients
print(f"Intercept (b0)          : {lr_model.intercept_:.4f}")
print(f"Coefficients [Hours, Att]: {lr_model.coef_.round(4)}")

# 5. Prediction and Evaluation
y_pred_lr = lr_model.predict(X_test_lr)
mae = mean_absolute_error(y_test_lr, y_pred_lr)
mse = mean_squared_error(y_test_lr, y_pred_lr)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_lr, y_pred_lr)

print("\nModel Evaluation Metrics on Test Set:")
print(f"   Mean Absolute Error (MAE) : {mae:.4f}")
print(f"   Mean Squared Error (MSE)  : {mse:.4f}")
print(f"   Root Mean Squared (RMSE)  : {rmse:.4f}")
print(f"   R-squared (R2 Score)      : {r2:.4f} (Explains {r2*100:.1f}% variance)")


# =============================================================================
# TASK 52
# =============================================================================
"""
TASK 52: Machine Learning (Perform Logistic Regression)
Reference Links:
- https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
- https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python
Statement:
Implement Logistic Regression for classification using scikit-learn:
1. Load binary classification dataset
2. Scale features using StandardScaler
3. Train LogisticRegression model
4. Predict classes and class probabilities
5. Evaluate model using Confusion Matrix and Classification Report
Week: 5
"""
print("\n" + "=" * 70)
print("TASK 52: Machine Learning - Logistic Regression Implementation")
print("=" * 70)

# 1. Load Breast Cancer classification dataset
cancer = load_breast_cancer()
X_cancer = cancer.data[:, :6]  # Use first 6 key morphological features
y_cancer = cancer.target        # 0 = Malignant, 1 = Benign

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_cancer, y_cancer, test_size=0.25, random_state=42, stratify=y_cancer
)

# 2. Scaling
scaler = StandardScaler()
X_train_log = scaler.fit_transform(X_train_log)
X_test_log = scaler.transform(X_test_log)

# 3. Fit Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_log, y_train_log)

# 4. Predictions
y_pred_log = log_reg.predict(X_test_log)
acc_log = accuracy_score(y_test_log, y_pred_log)

print(f"Logistic Regression Accuracy: {acc_log * 100:.2f}%\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test_log, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test_log, y_pred_log, target_names=cancer.target_names))


# =============================================================================
# TASK 53
# =============================================================================
"""
TASK 53: Machine Learning (Exercise on Decision Tree)
Reference Link: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
Statement:
Implement Decision Tree Classification using scikit-learn:
1. Load dataset (Iris flower dataset)
2. Split dataset into training and testing sets
3. Train DecisionTreeClassifier (tuning max_depth)
4. Evaluate classification accuracy
5. Export decision tree rule structure
Week: 6
"""
print("\n" + "=" * 70)
print("TASK 53: Machine Learning - Decision Tree Classification")
print("=" * 70)

iris = load_iris()
X_dt, y_dt = iris.data, iris.target

X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
    X_dt, y_dt, test_size=0.2, random_state=42, stratify=y_dt
)

# Train Decision Tree with max_depth=3 to prevent overfitting
dt_classifier = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
dt_classifier.fit(X_train_dt, y_train_dt)

y_pred_dt = dt_classifier.predict(X_test_dt)
acc_dt = accuracy_score(y_test_dt, y_pred_dt)

print(f"Decision Tree Accuracy on Iris Test Set: {acc_dt * 100:.2f}%\n")

# Feature importances
print("Feature Importances:")
for name, imp in zip(iris.feature_names, dt_classifier.feature_importances_):
    print(f"   {name:<25}: {imp:.4f}")

# Tree Decision Rules
print("\nDecision Tree Structure / Text Rules:")
tree_rules = export_text(dt_classifier, feature_names=list(iris.feature_names))
print(tree_rules)


# =============================================================================
# TASK 54
# =============================================================================
"""
TASK 54: Machine Learning (Exercise on SVM - Support Vector Machine)
Reference Link: https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
Statement:
Implement Support Vector Machine (SVM) Classification using scikit-learn:
1. Train SVM with Linear Kernel
2. Train SVM with Radial Basis Function (RBF) Kernel
3. Compare classification accuracies and decision performance
Week: 6
"""
print("\n" + "=" * 70)
print("TASK 54: Machine Learning - Support Vector Machine (SVM)")
print("=" * 70)

# Using Iris dataset for kernel comparison
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_dt, y_train_dt)
pred_linear = svm_linear.predict(X_test_dt)
acc_linear = accuracy_score(y_test_dt, pred_linear)

svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_dt, y_train_dt)
pred_rbf = svm_rbf.predict(X_test_dt)
acc_rbf = accuracy_score(y_test_dt, pred_rbf)

print(f"1. SVM with Linear Kernel Accuracy : {acc_linear * 100:.2f}%")
print(f"2. SVM with RBF (Kernel) Accuracy  : {acc_rbf * 100:.2f}%")
print("\nConclusion: Non-linear RBF kernel maps features into higher dimensions")
print("to effectively separate complex boundaries, while Linear kernel is optimal for linearly separable data.")


# =============================================================================
# TASK 55
# =============================================================================
"""
TASK 55: Machine Learning (Exercise on Time Series Analysis)
Reference Link: https://www.dataquest.io/blog/tutorial-time-series-analysis-with-pandas
Statement:
Implement Time Series Analysis with Pandas:
1. Generate DateTime index using pd.date_range()
2. Create time series with trend and weekly seasonality
3. Resample data across different time intervals (Weekly / Monthly means)
4. Compute rolling statistics (7-day moving average)
5. Generate lag features for forecasting preparation
Week: 6
"""
print("\n" + "=" * 70)
print("TASK 55: Machine Learning - Time Series Analysis with Pandas")
print("=" * 70)

# 1. Generate 90 days of daily time-series data
dates = pd.date_range(start="2026-01-01", periods=90, freq="D")
np.random.seed(42)

# Trend + 7-day Weekly Seasonality + Random Noise
trend = np.linspace(100, 160, 90)
seasonality = 15 * np.sin(2 * np.pi * np.arange(90) / 7)
noise = np.random.normal(0, 3, 90)
consumption = trend + seasonality + noise

ts_df = pd.DataFrame({'Consumption_kWh': consumption}, index=dates)

# 2. Resample: Weekly Average
weekly_summary = ts_df.resample('W').mean()

# 3. Rolling 7-day average
ts_df['7_Day_Moving_Avg'] = ts_df['Consumption_kWh'].rolling(window=7).mean()

# 4. Lag feature (Yesterday's consumption)
ts_df['Lag_1_Day'] = ts_df['Consumption_kWh'].shift(1)

print("Time Series Data Snapshot (first 10 days with Moving Average & Lag):")
print(ts_df.head(10).round(2))

print("\nWeekly Resampled Average (First 4 weeks):")
print(weekly_summary.head(4).round(2))


# =============================================================================
# TASK 56
# =============================================================================
"""
TASK 56: Machine Learning (Demonstration of Neural Networks from Scratch)
Reference Link: https://www.analyticsvidhya.com/blog/2020/07/neural-networks-from-scratch-in-python-and-r
Statement:
Demonstrate an Artificial Neural Network built entirely from scratch using NumPy:
1. Architecture: Input Layer (2 inputs), Hidden Layer (4 neurons), Output Layer (1 output)
2. Activation: Sigmoid function and its derivative
3. Loss: Mean Squared Error / Binary Cross-Entropy
4. Forward Propagation (matrix transformations)
5. Backpropagation using Chain Rule gradient computation
6. Training on non-linear XOR truth table
Week: 7
"""
print("\n" + "=" * 70)
print("TASK 56: Neural Network Implemented From Scratch (NumPy)")
print("=" * 70)

# Activation function and derivative
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    # x is the output of sigmoid(z)
    return x * (1.0 - x)

# Dataset: The classic Non-Linear XOR Problem
# Single-layer perceptron CANNOT solve XOR; 2-layer Neural Network CAN!
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_xor = np.array([[0], [1], [1], [0]])

# Neural Network Architecture: 2 inputs -> 4 hidden neurons -> 1 output
input_nodes = 2
hidden_nodes = 4
output_nodes = 1
learning_rate = 0.5
epochs = 5000

# Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.uniform(size=(input_nodes, hidden_nodes))
bias_hidden = np.zeros((1, hidden_nodes))

weights_hidden_output = np.random.uniform(size=(hidden_nodes, output_nodes))
bias_output = np.zeros((1, output_nodes))

# Training Loop
for epoch in range(epochs):
    # --- Forward Propagation ---
    hidden_linear = np.dot(X_xor, weights_input_hidden) + bias_hidden
    hidden_activation = sigmoid(hidden_linear)

    output_linear = np.dot(hidden_activation, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_linear)

    # --- Compute Error & Loss ---
    error = y_xor - predicted_output
    loss = np.mean(error ** 2)

    # --- Backpropagation ---
    # Gradient at output layer
    d_output = error * sigmoid_derivative(predicted_output)

    # Gradient at hidden layer
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_activation)

    # --- Gradient Descent Weight Updates ---
    weights_hidden_output += hidden_activation.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X_xor.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1:>5}/{epochs} | MSE Loss: {loss:.6f}")

print("\n--- XOR Truth Table Predictions After Training ---")
for i in range(len(X_xor)):
    input_sample = X_xor[i]
    true_label = y_xor[i][0]
    pred_prob = predicted_output[i][0]
    pred_binary = 1 if pred_prob >= 0.5 else 0
    print(f"Input: {input_sample} -> Expected: {true_label} | Predicted: {pred_prob:.4f} (Class: {pred_binary})")

print("\nConclusion: Neural network successfully trained from scratch and resolved non-linear XOR boundary!")
print("=" * 70)


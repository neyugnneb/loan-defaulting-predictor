import os
import re
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, accuracy_score ) 
import joblib
import shap


# Reads the processed and cleaned data
df = pd.read_csv('../data/processed/accepted_clean.csv')
MODEL_PATH = "../models/randomized_forest_pd_model.joblib"

TARGET_COLUMN = "default_flag"

NUMERIC_COLUMNS = [
    "loan_amnt", "funded_amnt", "funded_amnt_inv",
    "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc"
]

CATEGORICAL_COLUMNS = [
    "purpose", "home_ownership",
    "verification_status", "application_type"
]

X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
Y = df[TARGET_COLUMN]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_COLUMNS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS)
])

# The actual prediction model
model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=15,              
        min_samples_split=50,
        min_samples_leaf=20,     
        max_features='sqrt',       
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# Model fitting
model.fit(X_train, Y_train)

# Evaluate on both train and test sets
print("\n" + "="*50)
print("TRAINING SET PERFORMANCE")
print("="*50)
y_train_pred = model.predict(X_train)
print(classification_report(Y_train, y_train_pred, target_names=['No Default', 'Default']))

print("\n" + "="*50)
print("TEST SET PERFORMANCE")
print("="*50)
y_test_pred = model.predict(X_test)
print(classification_report(Y_test, y_test_pred, target_names=['No Default', 'Default']))

# Predict probabilities and ROC
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, y_proba)
roc_auc = auc(fpr, tpr)

# ROC curve graph
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Test different thresholds to find the best balance
print("\n" + "="*50)
print("THRESHOLD TESTING")
print("="*50)
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    accuracy = accuracy_score(Y_test, y_pred_thresh)
    print(f"\n--- Threshold: {threshold} | Accuracy: {accuracy:.4f} ---")
    print(classification_report(Y_test, y_pred_thresh, target_names=['No Default', 'Default']))

# After reviewing results above, pick your optimal threshold
# For example, if 0.4 looks best:
optimal_threshold = 0.4  # Change this based on what you see
y_pred = (y_proba >= optimal_threshold).astype(int)
print(f"\n{'='*50}")
print(f"FINAL MODEL USING THRESHOLD: {optimal_threshold}")
print(f"{'='*50}")
accuracy = accuracy_score(Y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(Y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

# Saves the model
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

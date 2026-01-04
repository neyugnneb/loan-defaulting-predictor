import os
import re
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score ) 
import shap


# Reads the processed and cleaned data
df = pd.read_csv('../data/processed/accepted_clean.csv')

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

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42, stratify = Y)

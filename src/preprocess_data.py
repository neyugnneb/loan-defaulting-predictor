import os
import re

import pandas as pd
import matplotlib.pyplot as plt

filepath = '../data/raw/'
processed_path = '../data/processed/'

# Gets the folder paths for the data
# accepted and rejected
folders = os.listdir(filepath)

# Accepted loans data
acc_folder = filepath + [f for f in folders if 'accepted' in f][0]
accepted_fn = acc_folder + '/' + os.listdir(acc_folder)[0]

# Reads the files. This will take some time, both files are huge
acc_df = pd.read_csv(accepted_fn, low_memory=False)

pd.options.display.max_rows = 1000

DEFAULT_STATUSES = [
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Late (16-30 days)"
]

PAID_STATUSES = ["Fully Paid"]

def convert_status(x):
    if x in PAID_STATUSES:
        return 0
    elif x in DEFAULT_STATUSES:
        return 1
    else:
        return None  # Current / unresolved


# Column that indicates if a loan is fully paid or not
acc_df["default_flag"] = acc_df["loan_status"].apply(convert_status)
acc_df = acc_df[acc_df["default_flag"].notna()]
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

# The data we want - gets rid of useless columns
acc_df = acc_df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]]

print(acc_df.head().T)
print(acc_df.info())

# Clean data, drop duplicates and handle missing values
for col in NUMERIC_COLUMNS:
    acc_df.loc[:, col] = pd.to_numeric(acc_df[col], errors="coerce")
    acc_df.loc[:, col] = acc_df[col].fillna(acc_df[col].median())

for col in CATEGORICAL_COLUMNS:
    acc_df.loc[:, col] = acc_df[col].fillna("Unknown")

print(acc_df.isna().sum())

os.makedirs(processed_path, exist_ok=True)
acc_df.to_csv(processed_path + "accepted_clean.csv", index=False)
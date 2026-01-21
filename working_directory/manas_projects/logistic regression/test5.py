import pandas as pd
import numpy as np

# ==============================================================
# 1. LOAD + PREPROCESS DATA
# ==============================================================

df = pd.read_csv("/Users/satviksingh/Documents/manas_projects/logistic regression/crime_train.csv")

# Remove useless columns
df = df.drop(columns=["Unnamed: 0", "Num"])

# Convert Yes/No → 1/0
df["closed"] = df["closed"].map({"Yes": 1, "No": 0}).astype(int)

# Convert datetime
df["case_filed"] = pd.to_datetime(df["case_filed"], errors="coerce")
df["year"] = df["case_filed"].dt.year
df["month"] = df["case_filed"].dt.month
df["hour"]  = df["case_filed"].dt.hour
df = df.drop(columns=["case_filed"])

# One-hot encode categorical features
df = pd.get_dummies(df, columns=["city", "crime_description", "sex", "weapon", "domain"])

# Convert to numpy
X = df.drop(columns=["closed"]).values.astype(float)
y = df["closed"].values.reshape(-1, 1)

# ==============================================================
# 2. FEATURE SCALING (VERY IMPORTANT)
# ==============================================================

X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8   # prevent division by zero
X = (X - X_mean) / X_std

# Add bias term (column of 1s)
m = X.shape[0]
X = np.hstack([np.ones((m, 1)), X])

# ==============================================================
# 3. STABLE SIGMOID + LOSS
# ==============================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow
    return 1 / (1 + np.exp(-z))

def predict_proba(X, weights):
    return sigmoid(np.dot(X, weights))

def compute_loss(y, y_pred):
    m = len(y)
    eps = 1e-9
    return -(1/m) * np.sum(y*np.log(y_pred + eps) + (1-y)*np.log(1-y_pred + eps))

# ==============================================================
# 4. TRAINING FUNCTION (GRADIENT DESCENT)

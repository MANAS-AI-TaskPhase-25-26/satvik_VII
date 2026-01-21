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
# 2. FEATURE SCALING
# ==============================================================

X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X = (X - X_mean) / X_std

# Add bias term
m = X.shape[0]
X = np.hstack([np.ones((m, 1)), X])

# ==============================================================
# 3. SIGMOID + log loss LOSS
# ==============================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict_proba(X, weights):
    return sigmoid(np.dot(X, weights))

def compute_loss(y, y_pred):
    m = len(y)
    eps = 1e-9
    return -(1/m) * np.sum(
        y*np.log(y_pred + eps) + (1-y)*np.log(1-y_pred + eps)
    )

# ==============================================================
# 4. TRAINING FUNCTION
# ==============================================================

def train_logistic_regression(X, y, lr=0.02, epochs=10000000):
    m, n = X.shape
    weights = np.zeros((n, 1))

    for epoch in range(epochs):
        y_pred = predict_proba(X, weights)
        gradient = (1/m) * np.dot(X.T, (y_pred - y))
        weights -= lr * gradient

        # Print status every 1000 epochs
        if epoch % 5000 == 0:
            loss = compute_loss(y, y_pred)
            print(f"Epoch {epoch:6d} | Loss: {loss}")

    return weights

# ==============================================================
# 5. TRAIN MODEL
# ==============================================================

print("Starting training...")
weights = train_logistic_regression(X, y)
print("Training complete.")

# ==============================================================
# 6. FINAL PREDICTION + ACCURACY
# ==============================================================

def predict(X, weights, threshold=0.5):
    return (predict_proba(X, weights) >= threshold).astype(int)

y_pred = predict(X, weights)
accuracy = np.mean(y_pred == y)

final_loss = compute_loss(y, predict_proba(X, weights))

print("\n==================== FINAL RESULT ====================")
print(f"Final Loss:      {final_loss:.6f}")
print(f"Final Accuracy:  {accuracy:.6f}")
print("======================================================")

print("\nLearned Weights:")
for i, weight in enumerate(weights):
    print(f"Weight {i}: {weight}")
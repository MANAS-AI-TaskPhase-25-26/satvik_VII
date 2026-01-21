# train_crime_model.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# keep numerical warnings visible if truly problematic:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)

# ---------------------
# Config
# ---------------------
CSV_PATH = "/Users/satviksingh/Documents/manas_projects/logistic regression/crime_train.csv"  # <-- change to your CSV filename
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------
# Load
# ---------------------
df = pd.read_csv(CSV_PATH)

# Basic target cleaning: map expected strings to 0/1
# If closed has other values, adapt mapping accordingly
if df["closed"].dtype == object:
    df["closed"] = df["closed"].str.strip().map({"Yes": 1, "No": 0})
df["closed"] = pd.to_numeric(df["closed"], errors="coerce")

# drop rows where target is missing
df = df.dropna(subset=["closed"])

# ---------------------
# Quick data cleaning heuristics
# ---------------------
# Fix obviously bad numeric columns that may have stray values
for col in ["age", "area", "police_department"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Example: if you have negative police_department or garbage small negative floats,
# clamp or treat as NaN so imputer handles them:
if "police_department" in df.columns:
    df.loc[df["police_department"] < 0, "police_department"] = np.nan

# ---------------------
# Feature selection
# ---------------------
# Pick features from the screenshot/your earlier pipeline.
# Adjust this list to match your CSV columns exactly.
numeric_features = [c for c in ["age", "area", "police_department"] if c in df.columns]
categorical_features = [c for c in ["city", "crime_description", "sex", "weapon", "domain"] if c in df.columns]

X = df[numeric_features + categorical_features]
y = df["closed"].astype(int)

# ---------------------
# Train/test split
# ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ---------------------
# Preprocessing pipelines
# ---------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
], remainder="drop")

# ---------------------
# Models to try
# ---------------------

# 1) Regularized Logistic Regression (stable; avoid polynomial term causing overflow)
logreg_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(
        solver="saga", # robust / supports L1/L2 with large dim
        penalty="l2",
        max_iter=2000,
        class_weight="balanced",  # helpful if target is imbalanced
        random_state=RANDOM_STATE
    ))
])

# 2) Random Forest (nonlinear, stable)
rf_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ))
])

# ---------------------
# Hyperparameter grids (small, practical)
# ---------------------
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

logreg_param_grid = {
    "clf__C": [0.01, 0.1, 1.0],
    "clf__penalty": ["l2"],   # keep l2 to avoid extreme sparsity issues here
}

rf_param_grid = {
    "clf__n_estimators": [100, 250],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_leaf": [1, 4]
}

# ---------------------
# Grid search for each model
# ---------------------
print("Training Logistic Regression (regularized) with GridSearchCV...")
lg_cv = GridSearchCV(logreg_pipe, logreg_param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
lg_cv.fit(X_train, y_train)
print("Best logreg params:", lg_cv.best_params_)

print("\nTraining Random Forest with GridSearchCV...")
rf_cv = GridSearchCV(rf_pipe, rf_param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
rf_cv.fit(X_train, y_train)
print("Best rf params:", rf_cv.best_params_)

# ---------------------
# Evaluate best models on test set
# ---------------------
def evaluate_model(model, X_test, y_test, name="model"):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"\n=== Evaluation for {name} ===")
    print("Accuracy: {:.4f}".format(acc))
    print("Precision: {:.4f}".format(prec))
    print("Recall: {:.4f}".format(rec))
    print("F1 score: {:.4f}".format(f1))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification report:\n", classification_report(y_test, preds, zero_division=0))

print("\nEvaluating Logistic Regression (best)...")
evaluate_model(lg_cv.best_estimator_, X_test, y_test, name="LogisticRegression (best)")

print("\nEvaluating Random Forest (best)...")
evaluate_model(rf_cv.best_estimator_, X_test, y_test, name="RandomForest (best)")

# ---------------------
# Save best of two (by F1 on test)
# ---------------------
from sklearn.metrics import f1_score as f1
lg_f1 = f1(y_test, lg_cv.best_estimator_.predict(X_test))
rf_f1 = f1(y_test, rf_cv.best_estimator_.predict(X_test))

best_model = rf_cv.best_estimator_ if rf_f1 >= lg_f1 else lg_cv.best_estimator_
print("\nBest model on test by F1:", "RandomForest" if rf_f1 >= lg_f1 else "LogisticRegression")

# Optional: save model with joblib
import joblib
joblib.dump(best_model, "best_crime_model.joblib")
print("Saved best model to best_crime_model.joblib")

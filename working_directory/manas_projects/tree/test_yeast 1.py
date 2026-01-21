import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load the uploaded Dataset
# Based on your file, the headers are already present.
df = pd.read_csv('tree/yeast.csv')

# 2. Preprocessing
# In your file, 'name' contains the localization site (the target).
# The other columns (mcg, gvh, alm, mit, erl, pox, vac, nuc) are the features.
X = df.drop(['name'], axis=1)
y = df['name']

# Encode the target labels (e.g., 'CYT' -> 0, 'MIT' -> 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Model Initialization
# We use 'entropy' to calculate Information Gain for splits
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)

# 4. Robust Evaluation (5-Fold Stratified Cross-Validation)
# We use 5 folds because the smallest class 'ERL' has only 5 samples.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

f1_scores = cross_val_score(dt_model, X, y_encoded, cv=skf, scoring='f1_macro')
mcc_scores = []

for train_index, test_index in skf.split(X, y_encoded):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y_encoded[train_index], y_encoded[test_index]
    
    dt_model.fit(X_train_cv, y_train_cv)
    y_pred_cv = dt_model.predict(X_test_cv)
    mcc_scores.append(matthews_corrcoef(y_test_cv, y_pred_cv))

print(f"--- Yeast Dataset Performance (5-Fold CV) ---")
print(f"Average F1 Score (Macro): {np.mean(f1_scores):.4f}")
print(f"Average MCC: {np.mean(mcc_scores):.4f}\n")

# 5. Final Detailed Report
# We use 'stratify' to ensure the 20% test set gets a fair share of rare classes.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

dt_model.fit(X_train, y_train)
final_pred = dt_model.predict(X_test)

# Identify which classes are actually in the test set to avoid labeling errors
labels_in_test = np.unique(np.concatenate((y_test, final_pred)))
target_names_in_test = le.inverse_transform(labels_in_test)

print("--- Detailed Classification Report ---")
print(classification_report(
    y_test, 
    final_pred, 
    labels=labels_in_test, 
    target_names=target_names_in_test,
    zero_division=0
))
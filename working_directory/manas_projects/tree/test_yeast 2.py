import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load the Dataset
# Note: Use the correct path for your machine, e.g., 'tree/yeast.csv' or the absolute path
df = pd.read_csv('tree/yeast.csv') 

# 2. Preprocessing
# 'name' is the target column in your specific file
X = df.drop(['name'], axis=1)
y = df['name']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Model Initialization (WITH TUNING)
# Added max_depth and min_samples_split to improve performance on Yeast data
dt_model = DecisionTreeClassifier(
    criterion='entropy', 
    max_depth=7,           # Prevents the tree from becoming too complex
    min_samples_split=10,  # Ensures a split only happens if there's enough data
    random_state=42
)

# 4. Robust Evaluation (5-Fold Stratified CV)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

f1_scores = cross_val_score(dt_model, X, y_encoded, cv=skf, scoring='f1_macro')
mcc_scores = []

for train_index, test_index in skf.split(X, y_encoded):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y_encoded[train_index], y_encoded[test_index]
    
    dt_model.fit(X_train_cv, y_train_cv)
    y_pred_cv = dt_model.predict(X_test_cv)
    mcc_scores.append(matthews_corrcoef(y_test_cv, y_pred_cv))

print(f"--- Tuned Yeast Dataset Performance ---")
print(f"Average F1 Score (Macro): {np.mean(f1_scores):.4f}")
print(f"Average MCC: {np.mean(mcc_scores):.4f}\n")

# 5. Final Detailed Report
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

dt_model.fit(X_train, y_train)
final_pred = dt_model.predict(X_test)

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset 
# Ensure the path matches your folder structure
df = pd.read_csv('tree/yeast.csv') 
X = df.drop(['name'], axis=1)
y = df['name']

# 2. Encode the target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Setup robust 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Iterate through different depths to find the best one
depths = range(1, 100)
accuracy_scores = []

print("Testing different depths...")
for d in depths:
    model = DecisionTreeClassifier(
        criterion='entropy', 
        max_depth=d, 
        min_samples_split=10, 
        random_state=42
    )
    acc = cross_val_score(model, X, y_encoded, cv=skf, scoring='accuracy').mean()
    accuracy_scores.append(acc)
    print(f"Depth {d}: Accuracy = {acc:.4f}")

# 5. Find and use the best depth
best_depth = depths[np.argmax(accuracy_scores)]
print(f"\nBest Depth based on Accuracy: {best_depth}")

# 6. Initialize and Train the Final Tuned Model
dt_model = DecisionTreeClassifier(
    criterion='entropy', 
    max_depth=best_depth,           
    min_samples_split=10,  
    class_weight='balanced',  # ADDED COMMA HERE
    random_state=42
)

# 7. Evaluate with Cross-Validation
f1_scores = cross_val_score(dt_model, X, y_encoded, cv=skf, scoring='f1_macro')
mcc_scores = []

for train_index, test_index in skf.split(X, y_encoded):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y_encoded[train_index], y_encoded[test_index]
    
    dt_model.fit(X_train_cv, y_train_cv)
    y_pred_cv = dt_model.predict(X_test_cv)
    mcc_scores.append(matthews_corrcoef(y_test_cv, y_pred_cv))

print(f"\n--- Tuned Yeast Dataset Performance (Balanced) ---")
print(f"Average F1 Score (Macro): {np.mean(f1_scores):.4f}")
print(f"Average MCC: {np.mean(mcc_scores):.4f}")

# 8. Detailed Classification Report
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

dt_model.fit(X_train, y_train)
final_pred = dt_model.predict(X_test)

labels_in_test = np.unique(np.concatenate((y_test, final_pred)))
target_names_in_test = le.inverse_transform(labels_in_test)

print("\n--- Detailed Classification Report ---")
print(classification_report(
    y_test, 
    final_pred, 
    labels=labels_in_test, 
    target_names=target_names_in_test,
    zero_division=0
))
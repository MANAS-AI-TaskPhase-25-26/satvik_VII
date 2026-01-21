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

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score

baseline_accuracy = np.mean(accuracy_scores)
baseline_f1 = np.mean(f1_scores)
baseline_mcc = np.mean(mcc_scores)

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score

baseline_accuracy = np.mean(accuracy_scores)
baseline_f1 = np.mean(f1_scores)
baseline_mcc = np.mean(mcc_scores)

# ============================================================
# BOOTSTRAPPING ONLY (NO BAGGING)
# ============================================================

bootstrap_true = []
bootstrap_pred = []

for train_idx, test_idx in skf.split(X, y_encoded):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # ---- BOOTSTRAP SAMPLE ----
    bootstrap_idx = np.random.choice(
        len(X_train),
        size=len(X_train),
        replace=True
    )

    X_boot = X_train.iloc[bootstrap_idx]
    y_boot = y_train[bootstrap_idx]

    tree = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=best_depth,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )

    tree.fit(X_boot, y_boot)
    preds = tree.predict(X_test)

    bootstrap_true.extend(y_test)
    bootstrap_pred.extend(preds)

# ---- METRICS ----
bootstrap_accuracy = accuracy_score(bootstrap_true, bootstrap_pred)
bootstrap_f1 = f1_score(bootstrap_true, bootstrap_pred, average='macro')
bootstrap_mcc = matthews_corrcoef(bootstrap_true, bootstrap_pred)

print("\n--- Bootstrapping Only Performance ---")
print(f"Accuracy: {bootstrap_accuracy:.4f}")
print(f"F1 Macro: {bootstrap_f1:.4f}")
print(f"MCC: {bootstrap_mcc:.4f}")

# ============================================================
# BOOTSTRAPPING ONLY (NO BAGGING)
# ============================================================

bootstrap_true = []
bootstrap_pred = []

for train_idx, test_idx in skf.split(X, y_encoded):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # ---- BOOTSTRAP SAMPLE ----
    bootstrap_idx = np.random.choice(
        len(X_train),
        size=len(X_train),
        replace=True
    )

    X_boot = X_train.iloc[bootstrap_idx]
    y_boot = y_train[bootstrap_idx]

    tree = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=best_depth,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )

    tree.fit(X_boot, y_boot)
    preds = tree.predict(X_test)

    bootstrap_true.extend(y_test)
    bootstrap_pred.extend(preds)

# ---- METRICS ----
bootstrap_accuracy = accuracy_score(bootstrap_true, bootstrap_pred)
bootstrap_f1 = f1_score(bootstrap_true, bootstrap_pred, average='macro')
bootstrap_mcc = matthews_corrcoef(bootstrap_true, bootstrap_pred)

print("\n--- Bootstrapping Only Performance ---")
print(f"Accuracy: {bootstrap_accuracy:.4f}")
print(f"F1 Macro: {bootstrap_f1:.4f}")
print(f"MCC: {bootstrap_mcc:.4f}")

# ============================================================
# BAGGING (BOOTSTRAP + ENSEMBLE)
# ============================================================

bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        criterion='entropy',
        max_depth=best_depth,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    ),
    n_estimators=50,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

# Accuracy
bagging_accuracy = cross_val_score(
    bagging_model, X, y_encoded, cv=skf, scoring='accuracy'
).mean()

# F1 Macro
bagging_f1 = cross_val_score(
    bagging_model, X, y_encoded, cv=skf, scoring='f1_macro'
).mean()

# MCC
bagging_mcc_scores = []
for train_idx, test_idx in skf.split(X, y_encoded):
    bagging_model.fit(X.iloc[train_idx], y_encoded[train_idx])
    preds = bagging_model.predict(X.iloc[test_idx])
    bagging_mcc_scores.append(
        matthews_corrcoef(y_encoded[test_idx], preds)
    )

bagging_mcc = np.mean(bagging_mcc_scores)

print("\n--- Bagging Performance ---")
print(f"Accuracy: {bagging_accuracy:.4f}")
print(f"F1 Macro: {bagging_f1:.4f}")
print(f"MCC: {bagging_mcc:.4f}")

# ============================================================
# PERFORMANCE COMPARISON PLOT (3 METHODS)
# ============================================================

methods = ['Decision Tree', 'Bootstrap Only', 'Bagging']
accuracy_vals = [
    baseline_accuracy,
    bootstrap_accuracy,
    bagging_accuracy
]
f1_vals = [
    baseline_f1,
    bootstrap_f1,
    bagging_f1
]
mcc_vals = [
    baseline_mcc,
    bootstrap_mcc,
    bagging_mcc
]

x = np.arange(len(methods))
width = 0.25

plt.figure()
plt.bar(x - width, accuracy_vals, width, label='Accuracy')
plt.bar(x, f1_vals, width, label='F1 Macro')
plt.bar(x + width, mcc_vals, width, label='MCC')

plt.xticks(x, methods)
plt.ylabel('Score')
plt.title('Decision Tree vs Bootstrapping vs Bagging')
plt.legend()
plt.show()


from sklearn.ensemble import AdaBoostClassifier

# ============================================================
# ============================================================
# BOOSTING (ADABOOST)
# ============================================================

boosting_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        criterion='entropy',
        max_depth=2,              # shallow trees are key
        min_samples_split=10,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

# Accuracy
boosting_accuracy = cross_val_score(
    boosting_model, X, y_encoded, cv=skf, scoring='accuracy'
).mean()

# F1 Macro
boosting_f1 = cross_val_score(
    boosting_model, X, y_encoded, cv=skf, scoring='f1_macro'
).mean()

# MCC (manual loop)
boosting_mcc_scores = []

for train_idx, test_idx in skf.split(X, y_encoded):
    boosting_model.fit(X.iloc[train_idx], y_encoded[train_idx])
    preds = boosting_model.predict(X.iloc[test_idx])
    boosting_mcc_scores.append(
        matthews_corrcoef(y_encoded[test_idx], preds)
    )

boosting_mcc = np.mean(boosting_mcc_scores)

print("\n--- Boosting (AdaBoost) Performance ---")
print(f"Accuracy: {boosting_accuracy:.4f}")
print(f"F1 Macro: {boosting_f1:.4f}")
print(f"MCC: {boosting_mcc:.4f}")

# ============================================================
# PERFORMANCE COMPARISON PLOT (4 METHODS)
# ============================================================

methods = [
    'Decision Tree',
    'Bootstrap Only',
    'Bagging',
    'Boosting'
]

accuracy_vals = [
    baseline_accuracy,
    bootstrap_accuracy,
    bagging_accuracy,
    boosting_accuracy
]

f1_vals = [
    baseline_f1,
    bootstrap_f1,
    bagging_f1,
    boosting_f1
]

mcc_vals = [
    baseline_mcc,
    bootstrap_mcc,
    bagging_mcc,
    boosting_mcc
]

x = np.arange(len(methods))
width = 0.25

plt.figure()
plt.bar(x - width, accuracy_vals, width, label='Accuracy')
plt.bar(x, f1_vals, width, label='F1 Macro')
plt.bar(x + width, mcc_vals, width, label='MCC')

plt.xticks(x, methods)
plt.ylabel('Score')
plt.title('Decision Tree vs Bootstrap vs Bagging vs Boosting')
plt.legend()
plt.show()


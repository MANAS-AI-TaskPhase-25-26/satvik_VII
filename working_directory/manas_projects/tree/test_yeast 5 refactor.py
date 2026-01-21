import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    classification_report
)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv('tree/yeast.csv')
X = df.drop(['name'], axis=1)
y = df['name']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 2. TUNE DECISION TREE DEPTH (USING ACCURACY)
# ============================================================

depths = range(1, 40)
accuracy_scores = []

print("Tuning Decision Tree depth...")
for d in depths:
    model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=d,
        min_samples_split=10,
        random_state=42
    )
    acc = cross_val_score(
        model, X, y_encoded,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1
    ).mean()
    accuracy_scores.append(acc)

best_depth = depths[np.argmax(accuracy_scores)]
print(f"Best Depth: {best_depth}")

# ============================================================
# 3. BASELINE DECISION TREE
# ============================================================

dt_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=best_depth,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)

dt_acc, dt_f1, dt_mcc = [], [], []

for train_idx, test_idx in skf.split(X, y_encoded):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    dt_model.fit(X_train, y_train)
    preds = dt_model.predict(X_test)

    dt_acc.append(accuracy_score(y_test, preds))
    dt_f1.append(f1_score(y_test, preds, average='macro'))
    dt_mcc.append(matthews_corrcoef(y_test, preds))

baseline_accuracy = np.mean(dt_acc)
baseline_f1 = np.mean(dt_f1)
baseline_mcc = np.mean(dt_mcc)

print("\n--- Decision Tree Performance ---")
print(f"Accuracy: {baseline_accuracy:.4f}")
print(f"F1 Macro: {baseline_f1:.4f}")
print(f"MCC: {baseline_mcc:.4f}")

# ============================================================
# 4. BOOTSTRAP ONLY (SINGLE TREE)
# ============================================================

boot_acc, boot_f1, boot_mcc = [], [], []

for train_idx, test_idx in skf.split(X, y_encoded):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    boot_idx = np.random.choice(len(X_train), len(X_train), replace=True)
    X_boot = X_train.iloc[boot_idx]
    y_boot = y_train[boot_idx]

    model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=best_depth,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_boot, y_boot)
    preds = model.predict(X_test)

    boot_acc.append(accuracy_score(y_test, preds))
    boot_f1.append(f1_score(y_test, preds, average='macro'))
    boot_mcc.append(matthews_corrcoef(y_test, preds))

bootstrap_accuracy = np.mean(boot_acc)
bootstrap_f1 = np.mean(boot_f1)
bootstrap_mcc = np.mean(boot_mcc)

print("\n--- Bootstrap Only Performance ---")
print(f"Accuracy: {bootstrap_accuracy:.4f}")
print(f"F1 Macro: {bootstrap_f1:.4f}")
print(f"MCC: {bootstrap_mcc:.4f}")

# ============================================================
# 5. BAGGING
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

bag_acc, bag_f1, bag_mcc = [], [], []

for train_idx, test_idx in skf.split(X, y_encoded):
    bagging_model.fit(X.iloc[train_idx], y_encoded[train_idx])
    preds = bagging_model.predict(X.iloc[test_idx])

    bag_acc.append(accuracy_score(y_encoded[test_idx], preds))
    bag_f1.append(f1_score(y_encoded[test_idx], preds, average='macro'))
    bag_mcc.append(matthews_corrcoef(y_encoded[test_idx], preds))

bagging_accuracy = np.mean(bag_acc)
bagging_f1 = np.mean(bag_f1)
bagging_mcc = np.mean(bag_mcc)

print("\n--- Bagging Performance ---")
print(f"Accuracy: {bagging_accuracy:.4f}")
print(f"F1 Macro: {bagging_f1:.4f}")
print(f"MCC: {bagging_mcc:.4f}")

# ============================================================
# 6. BOOSTING (ADABOOST)
# ============================================================

boosting_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

boost_acc, boost_f1, boost_mcc = [], [], []

for train_idx, test_idx in skf.split(X, y_encoded):
    boosting_model.fit(X.iloc[train_idx], y_encoded[train_idx])
    preds = boosting_model.predict(X.iloc[test_idx])

    boost_acc.append(accuracy_score(y_encoded[test_idx], preds))
    boost_f1.append(f1_score(y_encoded[test_idx], preds, average='macro'))
    boost_mcc.append(matthews_corrcoef(y_encoded[test_idx], preds))

boosting_accuracy = np.mean(boost_acc)
boosting_f1 = np.mean(boost_f1)
boosting_mcc = np.mean(boost_mcc)

print("\n--- Boosting Performance ---")
print(f"Accuracy: {boosting_accuracy:.4f}")
print(f"F1 Macro: {boosting_f1:.4f}")
print(f"MCC: {boosting_mcc:.4f}")

# ============================================================
# 7. RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=best_depth,
    min_samples_split=10,
    class_weight='balanced',
    bootstrap=True,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

rf_acc, rf_f1, rf_mcc = [], [], []

for train_idx, test_idx in skf.split(X, y_encoded):
    rf_model.fit(X.iloc[train_idx], y_encoded[train_idx])
    preds = rf_model.predict(X.iloc[test_idx])

    rf_acc.append(accuracy_score(y_encoded[test_idx], preds))
    rf_f1.append(f1_score(y_encoded[test_idx], preds, average='macro'))
    rf_mcc.append(matthews_corrcoef(y_encoded[test_idx], preds))

rf_accuracy = np.mean(rf_acc)
rf_f1 = np.mean(rf_f1)
rf_mcc = np.mean(rf_mcc)

print("\n--- Random Forest Performance ---")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"F1 Macro: {rf_f1:.4f}")
print(f"MCC: {rf_mcc:.4f}")

# ============================================================
# 8. FINAL COMPARISON PLOT
# ============================================================

methods = [
    'Decision Tree',
    'Bootstrap Only',
    'Bagging',
    'Boosting',
    'Random Forest'
]

accuracy_vals = [
    baseline_accuracy,
    bootstrap_accuracy,
    bagging_accuracy,
    boosting_accuracy,
    rf_accuracy
]

f1_vals = [
    baseline_f1,
    bootstrap_f1,
    bagging_f1,
    boosting_f1,
    rf_f1
]

mcc_vals = [
    baseline_mcc,
    bootstrap_mcc,
    bagging_mcc,
    boosting_mcc,
    rf_mcc
]

x = np.arange(len(methods))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, accuracy_vals, width, label='Accuracy')
plt.bar(x, f1_vals, width, label='F1 Macro')
plt.bar(x + width, mcc_vals, width, label='MCC')

plt.xticks(x, methods, rotation=15)
plt.ylabel('Score')
plt.title('Decision Tree vs Bootstrap vs Bagging vs Boosting vs Random Forest')
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("tree/yeast.csv")
X = df.drop(columns=["name"])
y = LabelEncoder().fit_transform(df["name"])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
num_classes = len(np.unique(y))

# ============================================================
# GENERIC EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X, y, cv):
    acc, f1, mcc = [], [], []
    cm_total = np.zeros((num_classes, num_classes))

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc.append(accuracy_score(y_test, preds))
        f1.append(f1_score(y_test, preds, average="macro"))
        mcc.append(matthews_corrcoef(y_test, preds))
        cm_total += confusion_matrix(y_test, preds)

    return np.mean(acc), np.mean(f1), np.mean(mcc), cm_total

# ============================================================
# HYPERPARAMETER TUNING (DEPTH 1–20)
# ============================================================

def tune_depth(model_fn, depths):
    scores = []
    for d in depths:
        model = model_fn(d)
        score = cross_val_score(
            model, X, y,
            cv=skf,
            scoring="f1_macro",
            n_jobs=-1
        ).mean()
        scores.append(score)
    return depths[np.argmax(scores)]

# ============================================================
# MODEL FACTORIES
# ============================================================

def decision_tree(depth):
    return DecisionTreeClassifier(
        criterion="entropy",
        max_depth=depth,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42
    )

def bagging(depth):
    return BaggingClassifier(
        estimator=decision_tree(depth),
        n_estimators=50,
        n_jobs=-1,
        random_state=42
    )

def boosting(depth):
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth),
        n_estimators=50,
        learning_rate=0.5,
        random_state=42
    )

def random_forest(depth):
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=depth,
        min_samples_split=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

# ============================================================
# RUN ALL MODELS
# ============================================================

models = {
    "Decision Tree": decision_tree,
    "Bagging": bagging,
    "Boosting": boosting,
    "Random Forest": random_forest
}

depth_range = range(1, 21)
results = {}

for name, model_fn in models.items():
    print(f"\nTuning {name}...")
    best_depth = tune_depth(model_fn, depth_range)
    print(f"Best depth: {best_depth}")

    model = model_fn(best_depth)
    acc, f1, mcc, cm = evaluate_model(model, X, y, skf)

    results[name] = {
        "accuracy": acc,
        "f1": f1,
        "mcc": mcc,
        "cm": cm
    }

# ============================================================
# PRINT RESULTS + CONFUSION MATRICES
# ============================================================

for name, res in results.items():
    print(f"\n--- {name} ---")
    print(f"Accuracy : {res['accuracy']:.4f}")
    print(f"F1 Macro : {res['f1']:.4f}")
    print(f"MCC      : {res['mcc']:.4f}")
    print("Confusion Matrix:")
    print(res["cm"].astype(int))

# ============================================================
# COMPARISON PLOT
# ============================================================

labels = list(results.keys())
accuracy = [results[m]["accuracy"] for m in labels]
f1 = [results[m]["f1"] for m in labels]
mcc = [results[m]["mcc"] for m in labels]

x = np.arange(len(labels))
w = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - w, accuracy, w, label="Accuracy")
plt.bar(x, f1, w, label="F1 Macro")
plt.bar(x + w, mcc, w, label="MCC")

plt.xticks(x, labels, rotation=15)
plt.ylabel("Score")
plt.title("Classifier Comparison")
plt.legend()
plt.tight_layout()
plt.show()

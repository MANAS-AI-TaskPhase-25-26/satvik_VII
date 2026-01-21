import numpy as np
import os
from PIL import Image
from collections import Counter

# =========================
# CONFIG
# =========================
IMG_SIZE = 32
LR = 0.01
EPOCHS = 500
HIDDEN = 128
VAL_SPLIT = 0.2
SEED = 42

np.random.seed(SEED)

# =========================
# UTILS
# =========================
def load_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img).reshape(-1) / 255.0

def one_hot(y, num_classes):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

# =========================
# LOAD DATA
# =========================
def load_train_data(root):
    X, y = [], []
    class_map = {}
    idx = 0

    for folder in sorted(os.listdir(root)):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue

        class_map[idx] = folder
        for img in os.listdir(folder_path):
            X.append(load_image(os.path.join(folder_path, img)))
            y.append(idx)
        idx += 1

    return np.array(X), np.array(y), class_map

def load_test_data(root):
    X, names = [], []
    for img in sorted(os.listdir(root)):
        X.append(load_image(os.path.join(root, img)))
        names.append(img)
    return np.array(X), names

# =========================
# ACTIVATIONS
# =========================
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# =========================
# LOSS
# =========================
def cross_entropy(y, y_hat):
    return -np.mean(np.sum(y * np.log(y_hat + 1e-9), axis=1))

# =========================
# MODEL INIT
# =========================
def init_params(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# =========================
# FORWARD
# =========================
def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_hat = softmax(z2)
    return z1, a1, y_hat

# =========================
# BACKWARD
# =========================
def backward(X, y, z1, a1, y_hat, W2, lr):
    m = X.shape[0]

    dz2 = y_hat - y
    dW2 = a1.T @ dz2 / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_derivative(z1)
    dW1 = X.T @ dz1 / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

# =========================
# METRICS
# =========================
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    num_classes = len(set(y_true))
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

# =========================
# TRAINING
# =========================
X, y, class_map = load_train_data("")
num_classes = len(class_map)
y_oh = one_hot(y, num_classes)

# Train/Val split
idx = np.random.permutation(len(X))
split = int(len(X) * (1 - VAL_SPLIT))
train_idx, val_idx = idx[:split], idx[split:]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y_oh[train_idx], y[val_idx]

W1, b1, W2, b2 = init_params(X.shape[1], HIDDEN, num_classes)

for epoch in range(EPOCHS):
    z1, a1, y_hat = forward(X_train, W1, b1, W2, b2)
    loss = cross_entropy(y_train, y_hat)

    dW1, db1, dW2, db2 = backward(X_train, y_train, z1, a1, y_hat, W2, LR)

    W1 -= LR * dW1
    b1 -= LR * db1
    W2 -= LR * dW2
    b2 -= LR * db2

    if epoch % 50 == 0:
        _, _, val_pred = forward(X_val, W1, b1, W2, b2)
        acc = accuracy(y_val, np.argmax(val_pred, axis=1))
        print(f"Epoch {epoch} | Loss {loss:.4f} | Val Acc {acc:.3f}")

# =========================
# CONFUSION MATRIX
# =========================
_, _, val_pred = forward(X_val, W1, b1, W2, b2)
cm = confusion_matrix(y_val, np.argmax(val_pred, axis=1))
print("\nConfusion Matrix:\n", cm)

# =========================
# PREDICT TIM'S KINS
# =========================
X_test, names = load_test_data("dataset/test")
_, _, test_pred = forward(X_test, W1, b1, W2, b2)
classes = np.argmax(test_pred, axis=1)

print("\nPredictions:")
for name, c in zip(names, classes):
    print(name, "->", class_map[c])

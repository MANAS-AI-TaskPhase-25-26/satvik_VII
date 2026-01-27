# hyperparam_search.py

import os
import numpy as np
from PIL import Image

# =====================================================
# PATHS
# =====================================================
TRAIN_PATH = "/Users/satviksingh/Documents/repos/satvik_VII/working_directory/neural net problems/Train"
TEST_PATH  = "/Users/satviksingh/Documents/repos/satvik_VII/working_directory/neural net problems/Test"

IMG_SIZE = 28
NUM_CLASSES = 5
HIDDEN_UNITS = 128

# =====================================================
# DATA LOADING
# =====================================================
def load_dataset(root_dir):
    X, Y = [], []

    class_names = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(root_dir, class_name)

        for file in os.listdir(class_path):
            if file.lower().endswith(".png"):
                path = os.path.join(class_path, file)

                img = Image.open(path).convert("L")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img = np.array(img).reshape(-1) / 255.0

                X.append(img)

                y = np.zeros(NUM_CLASSES)
                y[label] = 1
                Y.append(y)

    return np.array(X).T, np.array(Y).T

# =====================================================
# NN CORE
# =====================================================
def relu(Z):
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    dZ = dA.copy()
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    Z -= np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def init_params():
    W1 = np.random.randn(HIDDEN_UNITS, IMG_SIZE * IMG_SIZE) * 0.01
    b1 = np.zeros((HIDDEN_UNITS, 1))
    W2 = np.random.randn(NUM_CLASSES, HIDDEN_UNITS) * 0.01
    b2 = np.zeros((NUM_CLASSES, 1))
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return A2, (X, Z1, A1, Z2, A2)

def cross_entropy(A2, Y):
    m = Y.shape[1]
    return -np.sum(Y * np.log(A2 + 1e-8)) / m

def backward(cache, Y, W2):
    X, Z1, A1, _, A2 = cache
    m = X.shape[1]

    dZ2 = A2 - Y
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T @ dZ2
    dZ1 = relu_backward(dA1, Z1)
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def train(X, Y, lr, epochs):
    W1, b1, W2, b2 = init_params()

    for _ in range(epochs):
        A2, cache = forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(cache, Y, W2)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return W1, b1, W2, b2

def accuracy(X, Y, W1, b1, W2, b2):
    A2, _ = forward(X, W1, b1, W2, b2)
    return np.mean(np.argmax(A2, axis=0) == np.argmax(Y, axis=0))

# =====================================================
# HYPERPARAM SEARCH
# =====================================================
if __name__ == "__main__":
    X_train, Y_train = load_dataset(TRAIN_PATH)
    X_test, Y_test = load_dataset(TEST_PATH)

    learning_rates = [0.001, 0.01, 0.05, 0.1]
    epoch_options  = [500, 1000, 2000, 5000]

    best_acc = 0
    best_params = None

    for lr in learning_rates:
        for epochs in epoch_options:
            W1, b1, W2, b2 = train(X_train, Y_train, lr, epochs)
            acc = accuracy(X_test, Y_test, W1, b1, W2, b2)

            print(f"LR={lr} | EPOCHS={epochs} | TEST ACC={acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_params = (lr, epochs)

    with open("best_hyperparams.txt", "w") as f:
        f.write(f"learning_rate={best_params[0]}\n")
        f.write(f"epochs={best_params[1]}\n")
        f.write(f"test_accuracy={best_acc}\n")

    print("\nBEST FOUND:")
    print(best_params, "accuracy:", best_acc)

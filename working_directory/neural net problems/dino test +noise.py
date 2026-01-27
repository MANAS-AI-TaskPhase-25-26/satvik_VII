import os
import numpy as np
from PIL import Image

TRAIN_PATH = "/Users/satviksingh/Documents/repos/satvik_VII/working_directory/neural net problems/Train"
TEST_PATH  = "/Users/satviksingh/Documents/repos/satvik_VII/working_directory/neural net problems/Test"

IMG_SIZE = 28
NUM_CLASSES = 5
HIDDEN_UNITS = 8
LR = 0.2
EPOCHS = 15000

def load_dataset(root_dir):
    X, Y = [], []
    class_names = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))

    for label, cname in enumerate(class_names):
        for file in os.listdir(os.path.join(root_dir, cname)):
            if file.lower().endswith(".png"):
                img = Image.open(os.path.join(root_dir, cname, file)).convert("L")
                img = np.array(img.resize((IMG_SIZE, IMG_SIZE))).reshape(-1) / 255.0
                X.append(img)

                y = np.zeros(NUM_CLASSES)
                y[label] = 1
                Y.append(y)

    return np.array(X).T, np.array(Y).T

def add_noise(X, std=0.1):
    return np.clip(X + std * np.random.randn(*X.shape), 0.0, 1.0)

def relu(Z): return np.maximum(0, Z)

def relu_backward(dA, Z):
    dZ = dA.copy()
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    Z -= np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def init_params():
    W1 = np.random.randn(HIDDEN_UNITS, 784) * 0.01
    b1 = np.zeros((HIDDEN_UNITS, 1))
    W2 = np.random.randn(NUM_CLASSES, HIDDEN_UNITS) * 0.01
    b2 = np.zeros((NUM_CLASSES, 1))
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return A2, (X, Z1, A1, A2)

def backward(cache, Y, W2):
    X, Z1, A1, A2 = cache
    m = X.shape[1]

    dZ2 = A2 - Y
    dW2 = dZ2 @ A1.T / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dA1 = W2.T @ dZ2
    dZ1 = relu_backward(dA1, Z1)
    dW1 = dZ1 @ X.T / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return dW1, db1, dW2, db2

def train(X, Y):
    W1, b1, W2, b2 = init_params()

    for epoch in range(EPOCHS):
        A2, cache = forward(X, W1, b1, W2, b2)
        loss = -np.mean(np.sum(Y * np.log(A2 + 1e-8), axis=0))
        dW1, db1, dW2, db2 = backward(cache, Y, W2)

        W1 -= LR * dW1
        b1 -= LR * db1
        W2 -= LR * dW2
        b2 -= LR * db2

        if epoch % 500 == 0:
            print(f"epoch {epoch} | loss {loss:.4f}")

    return W1, b1, W2, b2

def accuracy(X, Y, W1, b1, W2, b2):
    A2, _ = forward(X, W1, b1, W2, b2)
    return np.mean(np.argmax(A2, axis=0) == np.argmax(Y, axis=0))

X_train, Y_train = load_dataset(TRAIN_PATH)
X_test, Y_test   = load_dataset(TEST_PATH)

X_train = add_noise(X_train, std=0.1)

W1, b1, W2, b2 = train(X_train, Y_train)

print("Train accuracy:", accuracy(X_train, Y_train, W1, b1, W2, b2))
print("Test accuracy :", accuracy(X_test, Y_test, W1, b1, W2, b2))

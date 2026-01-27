import os
import numpy as np
from PIL import Image

TRAIN_PATH = "/Users/satviksingh/Documents/repos/satvik_VII/working_directory/neural net problems/Train"
TEST_PATH  = "/Users/satviksingh/Documents/repos/satvik_VII/working_directory/neural net problems/Test"

IMG_SIZE = 28
NUM_CLASSES = 5
LR = 0.2
EPOCHS = 10000

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

def softmax(Z):
    Z -= np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def init_params():
    W = np.random.randn(NUM_CLASSES, 784) * 0.01
    b = np.zeros((NUM_CLASSES, 1))
    return W, b

def forward(X, W, b):
    Z = W @ X + b
    A = softmax(Z)
    return A, X

def backward(cache, Y, A):
    X = cache
    m = X.shape[1]
    dZ = A - Y
    dW = dZ @ X.T / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dW, db

def train(X, Y):
    W, b = init_params()

    for epoch in range(EPOCHS):
        A, cache = forward(X, W, b)
        loss = -np.mean(np.sum(Y * np.log(A + 1e-8), axis=0))
        dW, db = backward(cache, Y, A)

        W -= LR * dW
        b -= LR * db

        if epoch % 500 == 0:
            print(f"epoch {epoch} | loss {loss:.4f}")

    return W, b

def accuracy(X, Y, W, b):
    A, _ = forward(X, W, b)
    return np.mean(np.argmax(A, axis=0) == np.argmax(Y, axis=0))

X_train, Y_train = load_dataset(TRAIN_PATH)
X_test, Y_test   = load_dataset(TEST_PATH)

X_train = add_noise(X_train, std=0.1)

W, b = train(X_train, Y_train)

print("Train accuracy:", accuracy(X_train, Y_train, W, b))
print("Test accuracy :", accuracy(X_test, Y_test, W, b))

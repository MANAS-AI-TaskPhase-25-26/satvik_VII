# final_train.py

import os
import numpy as np
from PIL import Image

TRAIN_PATH = "/Users/satviksingh/Documents/repos/satvik_VII/working_directory/neural net problems/Train"
TEST_PATH  = "/Users/satviksingh/Documents/repos/satvik_VII/working_directory/neural net problems/Test"

IMG_SIZE = 28
NUM_CLASSES = 5
HIDDEN_UNITS = 128

# =====================================================
# LOAD HYPERPARAMS
# =====================================================
with open("best_hyperparams.txt") as f:
    lines = f.readlines()

LR = float(lines[0].split("=")[1])
EPOCHS = int(lines[1].split("=")[1])

# =====================================================
# DATA + NN (same as before)
# =====================================================
def load_dataset(root_dir):
    X, Y = [], []
    class_names = sorted(os.listdir(root_dir))

    for label, cname in enumerate(class_names):
        for file in os.listdir(os.path.join(root_dir, cname)):
            if file.endswith(".png"):
                img = Image.open(os.path.join(root_dir, cname, file)).convert("L")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                X.append(np.array(img).reshape(-1) / 255.0)

                y = np.zeros(NUM_CLASSES)
                y[label] = 1
                Y.append(y)

    return np.array(X).T, np.array(Y).T

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
    W1 = np.random.randn(HIDDEN_UNITS, IMG_SIZE*IMG_SIZE)*0.01
    b1 = np.zeros((HIDDEN_UNITS,1))
    W2 = np.random.randn(NUM_CLASSES, HIDDEN_UNITS)*0.01
    b2 = np.zeros((NUM_CLASSES,1))
    return W1,b1,W2,b2

def forward(X,W1,b1,W2,b2):
    A1 = relu(W1@X + b1)
    A2 = softmax(W2@A1 + b2)
    return A2, (X,A1)

def backward(cache,Y,W2,A2):
    X,A1 = cache
    m = X.shape[1]

    dZ2 = A2 - Y
    dW2 = dZ2@A1.T/m
    db2 = np.sum(dZ2,axis=1,keepdims=True)/m

    dA1 = W2.T@dZ2
    dZ1 = relu_backward(dA1,A1)
    dW1 = dZ1@X.T/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m

    return dW1,db1,dW2,db2

def accuracy(X,Y,W1,b1,W2,b2):
    A2,_ = forward(X,W1,b1,W2,b2)
    return np.mean(np.argmax(A2,0)==np.argmax(Y,0))

# =====================================================
# TRAIN
# =====================================================
X_train,Y_train = load_dataset(TRAIN_PATH)
X_test,Y_test   = load_dataset(TEST_PATH)

W1,b1,W2,b2 = init_params()

for epoch in range(EPOCHS):
    A2,cache = forward(X_train,W1,b1,W2,b2)
    dW1,db1,dW2,db2 = backward(cache,Y_train,W2,A2)

    W1 -= LR*dW1
    b1 -= LR*db1
    W2 -= LR*dW2
    b2 -= LR*db2

    if epoch % 200 == 0:
        print(f"epoch {epoch}")

print("\nFINAL RESULTS")
print("Train accuracy:", accuracy(X_train,Y_train,W1,b1,W2,b2))
print("Test accuracy :", accuracy(X_test,Y_test,W1,b1,W2,b2))

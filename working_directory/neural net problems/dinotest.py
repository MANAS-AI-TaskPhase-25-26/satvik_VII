'''
    importing data 
        puting it in arrays 
            there are 5 training classes 
                each class has 801 images 
                    dim of images are 28x28
                    images are greyscale
                    png  
            there are 5 test classes 
                each class has 201 images 



        what should the dimension of X,w,b be???
        what should the structure of the neural network be ??
            how many layers how many neurons ??
                        
    gernating w and b 

        dimensions so that i can take dot product of it all 
    
    forward prop 
        can this be done with shalow neural network ?
            only if not try other approach 

    storing result of forward prop 
        array dimensions 

    back prop 
        generating loss function 
        differentiation of final layer then subsequent layers 

        updating weights     


    
    
    testing accuracy 
        load other folder 
            test for percentage acuracy 

'''



import os
import numpy as np
from PIL import Image

# =====================================================
# CONFIG — CHANGE ONLY THESE TWO PATHS
# =====================================================
TRAIN_PATH = "/Users/satviksingh/Documents/repos/satvik_VII/working_directory/neural net problems/Train"
TEST_PATH  = "/Users/satviksingh/Documents/repos/satvik_VII/working_directory/neural net problems/Test"

IMG_SIZE = 28
NUM_CLASSES = 5
HIDDEN_UNITS = 128
LR = 0.2
EPOCHS = 3000

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

    X = np.array(X).T   # (784, m)
    Y = np.array(Y).T   # (5, m)
    return X, Y

# =====================================================
# ACTIVATIONS
# =====================================================
def relu(Z):
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    dZ = dA.copy()
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# =====================================================
# INITIALIZATION
# =====================================================
def init_params():
    W1 = np.random.randn(HIDDEN_UNITS, IMG_SIZE * IMG_SIZE) * 0.01
    b1 = np.zeros((HIDDEN_UNITS, 1))
    W2 = np.random.randn(NUM_CLASSES, HIDDEN_UNITS) * 0.01
    b2 = np.zeros((NUM_CLASSES, 1))
    return W1, b1, W2, b2

# =====================================================
# FORWARD
# =====================================================
def forward(X, W1, b1, W2, b2):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    cache = (X, Z1, A1, Z2, A2)
    return A2, cache

# =====================================================
# LOSS
# =====================================================
def cross_entropy(A2, Y):
    m = Y.shape[1]
    return -np.sum(Y * np.log(A2 + 1e-8)) / m

# =====================================================
# BACKPROP
# =====================================================
def backward(cache, Y, W2):
    X, Z1, A1, Z2, A2 = cache
    m = X.shape[1]

    dZ2 = A2 - Y
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T @ dZ2
    dZ1 = relu_backward(dA1, Z1)
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# =====================================================
# UPDATE
# =====================================================
def update(W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 -= LR * dW1
    b1 -= LR * db1
    W2 -= LR * dW2
    b2 -= LR * db2
    return W1, b1, W2, b2

# =====================================================
# TRAIN
# =====================================================
def train(X, Y):
    W1, b1, W2, b2 = init_params()

    for epoch in range(EPOCHS):
        A2, cache = forward(X, W1, b1, W2, b2)
        loss = cross_entropy(A2, Y)
        dW1, db1, dW2, db2 = backward(cache, Y, W2)
        W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2)

        if epoch % 100 == 0:
            print(f"epoch {epoch} | loss {loss:.4f}")

    return W1, b1, W2, b2

# =====================================================
# ACCURACY
# =====================================================
def accuracy(X, Y, W1, b1, W2, b2):
    A2, _ = forward(X, W1, b1, W2, b2)
    preds = np.argmax(A2, axis=0)
    labels = np.argmax(Y, axis=0)
    return np.mean(preds == labels)

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    if not os.path.isdir(TRAIN_PATH):
        raise FileNotFoundError(TRAIN_PATH)
    if not os.path.isdir(TEST_PATH):
        raise FileNotFoundError(TEST_PATH)

    X_train, Y_train = load_dataset(TRAIN_PATH)
    X_test, Y_test = load_dataset(TEST_PATH)

    print("Train:", X_train.shape, Y_train.shape)
    print("Test :", X_test.shape, Y_test.shape)

    W1, b1, W2, b2 = train(X_train, Y_train)

    print("Train accuracy:", accuracy(X_train, Y_train, W1, b1, W2, b2))
    print("Test accuracy :", accuracy(X_test, Y_test, W1, b1, W2, b2))

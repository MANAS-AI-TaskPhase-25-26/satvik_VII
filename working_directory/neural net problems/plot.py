import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =====================================================
# PATHS
# =====================================================
TRAIN_PATH = "/Users/satviksingh/Documents/repos/satvik_VII/working_directory/neural net problems/Train"

IMG_SIZE = 28
NUM_CLASSES = 5

# =====================================================
# DATA LOADING
# =====================================================
def load_dataset(root_dir):
    X, Y = [], []

    class_names = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )

    for label, cname in enumerate(class_names):
        class_path = os.path.join(root_dir, cname)

        for file in os.listdir(class_path):
            if file.lower().endswith(".png"):
                img = Image.open(os.path.join(class_path, file)).convert("L")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img = np.array(img).reshape(-1) / 255.0

                X.append(img)

                y = np.zeros(NUM_CLASSES)
                y[label] = 1
                Y.append(y)

    return np.array(X).T, np.array(Y).T

# =====================================================
# PCA VISUALIZATION
# =====================================================
def pca_2d_plot(X, Y):
    Xc = X - np.mean(X, axis=1, keepdims=True)

    C = np.cov(Xc)
    eigvals, eigvecs = np.linalg.eigh(C)

    W = eigvecs[:, -2:]
    X_pca = W.T @ Xc

    labels = np.argmax(Y, axis=0)

    plt.figure(figsize=(6, 6))
    for c in range(NUM_CLASSES):
        idx = labels == c
        plt.scatter(X_pca[0, idx], X_pca[1, idx], s=10, label=f"class {c}")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (2D) Projection")
    plt.legend()
    plt.tight_layout()

# =====================================================
# CLASS MEAN IMAGES
# =====================================================
def plot_class_means(X, Y):
    labels = np.argmax(Y, axis=0)

    plt.figure(figsize=(10, 2))
    for c in range(NUM_CLASSES):
        Xc = X[:, labels == c]
        mean_img = np.mean(Xc, axis=1).reshape(IMG_SIZE, IMG_SIZE)

        plt.subplot(1, NUM_CLASSES, c + 1)
        plt.imshow(mean_img, cmap="gray")
        plt.axis("off")
        plt.title(f"class {c}")

    plt.suptitle("Mean Image per Class")
    plt.tight_layout()

# =====================================================
# TRAIN LINEAR CLASSIFIER
# =====================================================
def softmax(Z):
    Z -= np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def train_linear(X, Y, lr=0.2, epochs=3000):
    W = np.random.randn(NUM_CLASSES, X.shape[0]) * 0.01
    b = np.zeros((NUM_CLASSES, 1))

    for _ in range(epochs):
        Z = W @ X + b
        A = softmax(Z)

        dZ = A - Y
        W -= lr * (dZ @ X.T) / X.shape[1]
        b -= lr * np.mean(dZ, axis=1, keepdims=True)

    return W, b

# =====================================================
# VISUALIZE LINEAR WEIGHTS
# =====================================================
def plot_linear_weights(W):
    plt.figure(figsize=(10, 2))
    for c in range(NUM_CLASSES):
        plt.subplot(1, NUM_CLASSES, c + 1)
        plt.imshow(W[c].reshape(IMG_SIZE, IMG_SIZE), cmap="seismic")
        plt.colorbar(fraction=0.046)
        plt.axis("off")
        plt.title(f"class {c}")

    plt.suptitle("Linear Classifier Weights")
    plt.tight_layout()

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    X_train, Y_train = load_dataset(TRAIN_PATH)

    pca_2d_plot(X_train, Y_train)
    plot_class_means(X_train, Y_train)

    W, b = train_linear(X_train, Y_train)
    plot_linear_weights(W)

    # 🔥 ONE blocking call only
    plt.show()
xqq
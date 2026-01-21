import numpy as np

# Training data (non-linear target)
x = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([[1], [4], [9], [16]], dtype=float)  # y = x^2

# Initialize weights
W1 = np.random.randn(1, 2)
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

lr = 0.01

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

# Training loop
for epoch in range(5000):
    # ---- Forward pass ----
    z1 = x @ W1 + b1
    h = relu(z1)
    y_pred = h @ W2 + b2

    loss = np.mean((y - y_pred) ** 2)

    # ---- Backprop ----
    dL_dy = 2 * (y_pred - y) / len(y)

    dW2 = h.T @ dL_dy
    db2 = np.sum(dL_dy, axis=0, keepdims=True)

    dh = dL_dy @ W2.T
    dz1 = dh * relu_grad(z1)

    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # ---- Update ----
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 500 == 0:
        print(f"epoch {epoch} | loss {loss:.4f}")

# Test
print("Prediction for x=5:", relu(np.array([[5]]) @ W1 + b1) @ W2 + b2)

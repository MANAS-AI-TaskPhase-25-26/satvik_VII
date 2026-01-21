import numpy as np

# Training data
x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([2, 4, 6, 8], dtype=float)

# Initialize parameters
w = 0.0
b = 0.0
lr = 0.01

# Training loop
for epoch in range(1000):
    # Forward pass
    y_pred = w * x + b

    # Loss (Mean Squared Error)
    loss = np.mean((y - y_pred) ** 2)

    # Gradients
    dw = -2 * np.mean(x * (y - y_pred))
    db = -2 * np.mean(y - y_pred)

    # Update
    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        print(f"epoch {epoch} | loss {loss:.4f}")

print("Learned weight:", w)
print("Learned bias:", b)

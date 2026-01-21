import numpy as np

# ------------------------------
# ACTIVATION FUNCTIONS
# ------------------------------

def sigmoid(x):
    """
    Squashes any real number into (0,1)
    Used for neuron activation
    """
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    """
    Derivative of sigmoid
    IMPORTANT:
    We pass the OUTPUT of sigmoid, not the input.
    Because: d(sigmoid)/dx = y * (1 - y)
    """
    return y * (1 - y)


# ------------------------------
# TRAINING DATA (XOR)
# ------------------------------

# Inputs: all binary combinations
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Outputs (XOR truth table)
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])


# ------------------------------
# NETWORK INITIALIZATION
# ------------------------------

# Fixing seed just for reproducibility
np.random.seed(0)

# W1: weights from input layer → hidden layer
# Shape: (2 inputs × 2 hidden neurons)
W1 = np.random.uniform(-1, 1, (2, 2))

# b1: bias for hidden layer (one per neuron)
b1 = np.random.uniform(-1, 1, (1, 2))

# W2: weights from hidden layer → output neuron
# Shape: (2 hidden neurons × 1 output)
W2 = np.random.uniform(-1, 1, (2, 1))

# b2: bias for output neuron
b2 = np.random.uniform(-1, 1, (1, 1))

# Learning rate
lr = 0.1


# ------------------------------
# TRAINING LOOP
# ------------------------------

for epoch in range(20000):          # repeat many times
    for i in range(4):              # loop over each data point

        # Take one input-output pair
        x = X[i:i+1]                # shape (1,2)
        y = Y[i:i+1]                # shape (1,1)

        # -------- FORWARD PASS --------

        # Hidden layer weighted sum
        z1 = np.dot(x, W1) + b1

        # Hidden layer activation
        h = sigmoid(z1)

        # Output neuron weighted sum
        z2 = np.dot(h, W2) + b2

        # Final output activation
        out = sigmoid(z2)

        # -------- ERROR --------

        # How far prediction is from truth
        # Using squared error loss derivative
        err = y - out

        # -------- BACKPROPAGATION --------

        # Gradient at output neuron
        # dL/dout * dout/dz
        d_out = err * dsigmoid(out)

        # Gradient flowing into hidden layer
        # Chain rule:
        # dL/dh = dL/dout * dout/dh
        d_h = d_out.dot(W2.T) * dsigmoid(h)

        # -------- WEIGHT UPDATES --------

        # Update hidden → output weights
        # h.T because we want (2x1)
        W2 += lr * h.T.dot(d_out)

        # Update output bias
        b2 += lr * d_out

        # Update input → hidden weights
        W1 += lr * x.T.dot(d_h)

        # Update hidden bias
        b1 += lr * d_h


# ------------------------------
# TESTING THE NETWORK
# ------------------------------

print("\nFinal predictions:\n")

for x in X:
    h = sigmoid(np.dot(x, W1) + b1)
    out = sigmoid(np.dot(h, W2) + b2)

    # Convert sigmoid output to 0 or 1
    print(x, "→", round(out[0][0], 3), "(binary:", 1 if out > 0.5 else 0, ")")

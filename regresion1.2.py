import pandas as pd


#initial data reading 
df = pd.read_csv("train.csv")
column_means = df.mean(numeric_only=True)
df = df.fillna(column_means)
print(df.head())

#data extraction 
X_train = df.drop(columns=["close","date","symbols"]).values#skiping non numeric data and target
y_train = df["close"].values



print(X_train.shape)#prints structure
print(y_train.shape)

print(X_train[:5]) #prints values  
print(y_train[:5])  


#plot the data

#normalize the data 
import numpy as np
mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)
X_train_norm = (X_train - mu) / sigma

#plot again

#function generation 
m, n = X_train_norm.shape
np.random.seed(1)
w = np.random.rand(n)
b = 0.0

#cost function generation 
def compute_cost(X, y, w, b):
    """
    Computes the cost (Mean Squared Error) for linear regression.
    """
    m = X.shape[0]
    # Prediction: f_wb = X * w + b (using NumPy dot product)
    f_wb = X @ w + b
    
    # Calculate squared error
    cost = (f_wb - y)**2
    
    # Return the mean cost (scaled by 1/2m for cleaner gradient)
    total_cost = np.sum(cost) / (2 * m)
    return total_cost

#gradient descent function generation
# --- 1. GRADIENT FUNCTION (The missing step to calculate the direction) ---
def compute_gradient(X, y, w, b):
    """
    Computes the gradient (partial derivatives) for w and b.
    """
    m = X.shape[0]
    f_wb = X @ w + b    # Predictions: X * w + b
    err = f_wb - y      # Prediction error: (predicted - actual)

    # Vectorized gradient for all weights (dw)
    # dw = X_T * err / m
    dw = (X.T @ err) / m

    # Gradient for bias (db)
    # db = Sum(err) / m
    db = np.sum(err) / m

    return dw, db

# --- 2. GRADIENT DESCENT LOOP (The missing step to iteratively update w and b) ---
def gradient_descent(X, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to learn w and b.
    """
    w = w_in
    b = b_in
    J_history = []  # To store cost history for later analysis

    for i in range(num_iters):
        # Calculate the gradient 
        dw, db = gradient_function(X, y, w, b)

        # Update rule: Simultaneously update w and b
        w = w - alpha * dw
        b = b - alpha * db

        # Record and print the cost
        cost = cost_function(X, y, w, b)
        J_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i:4d}: Cost {cost:.2f}")

    return w, b, J_history

# --- APPLY GRADIENT DESCENT ---
iterations = 3000

alpha = 0.01  # Learning rate

# Run the algorithm using your normalized data and defined parameters
w_final, b_final, J_hist = gradient_descent(
    X_train_norm, y_train, w, b, alpha, iterations, compute_cost, compute_gradient
)

print("\n--- Manual Gradient Descent Training Complete ---")
print(f"Optimal weights (w): {w_final}")
print(f"Optimal bias (b): {b_final:.2f}")

# --- DEMONSTRATE PREDICTION ---
def predict_manual(X, w, b):
    return X @ w + b

y_pred_manual = predict_manual(X_train_norm[:5], w_final, b_final)

print("\n--- Model Performance on First 5 Samples ---")
print(f"Actual Prices (y_train[:5]): \t{y_train[:5]}")
print(f"Predicted Prices: \t\t{y_pred_manual}")
# Set hyperparameters

# Plot cost history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(iterations), J_hist)
plt.title("Cost (MSE) vs. Iterations")
plt.xlabel("Iteration Number")
plt.ylabel("Cost (Mean Squared Error)")
plt.grid(True)
plt.show()

print("\nCost history plot displayed.")

# --- (Add this after your cost plot code) ---

print("Generating Actual vs. Predicted plot...")

# 1. Get predictions for ALL data
y_pred_full = predict_manual(X_train_norm, w_final, b_final)

# 2. Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_pred_full, alpha=0.5, label='Actual vs. Predicted')

# 3. Add a perfect-prediction line (y=x)
min_val = min(np.min(y_train), np.min(y_pred_full))
max_val = max(np.max(y_train), np.max(y_pred_full))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

plt.title('Model Performance: Actual vs. Predicted')
plt.xlabel('Actual Close Price')
plt.ylabel('Predicted Close Price')
plt.legend()
plt.grid(True)
plt.show()

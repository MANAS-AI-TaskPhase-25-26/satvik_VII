import pandas as panda
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


# ------------------ INSPECT ------------------
def inspect(var, name="variable"):
    print(f"Name : {name}")
    print(f"Type : {type(var)}")

    if hasattr(var, "shape"):
        print(f"Shape: {var.shape}")
    else:
        print("Shape: N/A")

    print("Value:")
    print(var)
    print("-" * 40)


# ------------------ SCALER ------------------
def scaler(x):
    x_scaled = np.zeros_like(x, dtype=float)

    for j in range(x.shape[1]):
        mean = np.mean(x[:, j])
        std = np.std(x[:, j])
        if std == 0:
            std = 1.0  # FIX: avoid divide by zero

        for i in range(x.shape[0]):
            x_scaled[i, j] = (x[i, j] - mean) / std

    return x_scaled


# ------------------ KMEANS INIT ------------------
def generate(z, x):
    i = 0
    while i <= 2:
        r = np.random.randint(0, x.shape[0])
        z[i] = x[r]
        i += 1
    return z


# ------------------ ASSIGN ------------------
def assign(z, x):
    y = np.zeros(x.shape[0], dtype=int)
    count_x = 0

    while count_x < x.shape[0]:
        z0 = np.linalg.norm(z[0] - x[count_x])
        z1 = np.linalg.norm(z[1] - x[count_x])
        z2 = np.linalg.norm(z[2] - x[count_x])

        if (z1 < z2) and (z1 < z0):
            y[count_x] = 1
        elif (z0 < z2) and (z0 < z1):
            y[count_x] = 0
        else:
            y[count_x] = 2

        count_x += 1

    return y


# ------------------ CENTROID UPDATE ------------------
def cluster_centroid(x, y):
    cluster_0 = np.array([], dtype=int)
    cluster_1 = np.array([], dtype=int)
    cluster_2 = np.array([], dtype=int)

    counter = 0
    while counter < x.shape[0]:
        if y[counter] == 0:
            cluster_0 = np.append(cluster_0, counter)
        elif y[counter] == 1:
            cluster_1 = np.append(cluster_1, counter)
        elif y[counter] == 2:
            cluster_2 = np.append(cluster_2, counter)
        counter += 1

    # FIX: handle empty clusters safely
    m0 = np.mean(x[cluster_0], axis=0) if len(cluster_0) > 0 else x[np.random.randint(len(x))]
    m1 = np.mean(x[cluster_1], axis=0) if len(cluster_1) > 0 else x[np.random.randint(len(x))]
    m2 = np.mean(x[cluster_2], axis=0) if len(cluster_2) > 0 else x[np.random.randint(len(x))]

    z = np.array([m0, m1, m2])
    return z


# ------------------ LOAD DATA ------------------
data = panda.read_csv(
    "/Users/satviksingh/Documents/manas_projects/classification/seeds_data.csv"
)

x_loader = data.drop(['Class'], axis=1)
y_loader = data['Class']

x = x_loader.values
y_true = y_loader.values  # true labels: 1,2,3

x = scaler(x)


# ------------------ RUN KMEANS ------------------
z = np.zeros((3, x.shape[1]))
z = generate(z, x)

counter = 0
while counter <= 10:
    y = assign(z, x)
    z = cluster_centroid(x, y)
    counter += 1


# ------------------ CONFUSION MATRIX (UNALIGNED) ------------------
cm = confusion_matrix(y_true, y)


# ------------------ ALIGN CLUSTERS (HUNGARIAN) ------------------
row_ind, col_ind = linear_sum_assignment(-cm)

label_map = {col: row for row, col in zip(row_ind, col_ind)}

y_aligned = np.zeros_like(y)
for i in range(len(y)):
    y_aligned[i] = label_map[y[i]]


# ------------------ CONFUSION MATRIX (ALIGNED) ------------------
cm_aligned = confusion_matrix(y_true, y_aligned)


# ------------------ ACCURACY ------------------
accuracy = np.trace(cm_aligned) / np.sum(cm_aligned)
print("Clustering Accuracy:", accuracy)

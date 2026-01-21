import pandas as panda
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA


# ------------------ SCALER ------------------
def scaler(x):
    x_scaled = np.zeros_like(x, dtype=float)
    for j in range(x.shape[1]):
        mean = np.mean(x[:, j])
        std = np.std(x[:, j])
        if std == 0:
            std = 1.0
        for i in range(x.shape[0]):
            x_scaled[i, j] = (x[i, j] - mean) / std
    return x_scaled


# ------------------ KMEANS ------------------
def generate(z, x):
    i = 0
    while i < z.shape[0]:
        r = np.random.randint(0, x.shape[0])
        z[i] = x[r]
        i += 1
    return z


def assign(z, x):
    y = np.zeros(x.shape[0], dtype=int)
    count_x = 0

    while count_x < x.shape[0]:
        d0 = np.linalg.norm(z[0] - x[count_x])
        d1 = np.linalg.norm(z[1] - x[count_x])
        d2 = np.linalg.norm(z[2] - x[count_x])

        if d1 < d0 and d1 < d2:
            y[count_x] = 1
        elif d0 < d1 and d0 < d2:
            y[count_x] = 0
        else:
            y[count_x] = 2

        count_x += 1
    return y


def cluster_centroid(x, y):
    z = np.zeros((3, x.shape[1]))
    for k in range(3):
        pts = x[y == k]
        z[k] = np.mean(pts, axis=0) if len(pts) > 0 else x[np.random.randint(len(x))]
    return z


def inertia(x, y, z):
    return sum(np.linalg.norm(x[i] - z[y[i]])**2 for i in range(len(x)))


# ------------------ LOAD DATA ------------------
data = panda.read_csv(
    "/Users/satviksingh/Documents/manas_projects/classification/seeds_data.csv"
)

x = scaler(data.drop(['Class'], axis=1).values)
y_true = data['Class'].values


# =====================================================
# 1️⃣ KNEE (ELBOW) CURVE
# =====================================================
Ks = range(1, 7)
inertias_K = []

for K in Ks:
    z = np.zeros((K, x.shape[1]))
    z = generate(z, x)

    for _ in range(10):
        # temporary assign for arbitrary K
        y_tmp = np.zeros(len(x), dtype=int)
        for i in range(len(x)):
            d = [np.linalg.norm(x[i] - z[k]) for k in range(K)]
            y_tmp[i] = np.argmin(d)

        for k in range(K):
            pts = x[y_tmp == k]
            z[k] = np.mean(pts, axis=0) if len(pts) > 0 else x[np.random.randint(len(x))]

    inertias_K.append(
        sum(np.linalg.norm(x[i] - z[y_tmp[i]])**2 for i in range(len(x)))
    )

plt.figure()
plt.plot(Ks, inertias_K, marker='o')
plt.title("Elbow (Knee) Curve")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Loss)")
plt.show()


# =====================================================
# 2️⃣ FINAL K = 3 + LOSS vs EPOCHS
# =====================================================
K = 3
z = np.zeros((3, x.shape[1]))
z = generate(z, x)

loss_per_epoch = []

epoch = 0
while epoch <= 10:
    y = assign(z, x)
    z = cluster_centroid(x, y)
    loss_per_epoch.append(inertia(x, y, z))
    epoch += 1

plt.figure()
plt.plot(range(len(loss_per_epoch)), loss_per_epoch, marker='o')
plt.title("Loss (Inertia) vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Inertia")
plt.show()


# =====================================================
# 3️⃣ CONFUSION MATRIX + ACCURACY
# =====================================================
cm = confusion_matrix(y_true, y)

row_ind, col_ind = linear_sum_assignment(-cm)
label_map = {col: row for row, col in zip(row_ind, col_ind)}

y_aligned = np.array([label_map[i] for i in y])
cm_aligned = confusion_matrix(y_true, y_aligned)

plt.figure()
plt.imshow(cm_aligned)
plt.colorbar()
plt.title("Confusion Matrix (Aligned)")
plt.xlabel("Predicted Cluster")
plt.ylabel("True Label")
plt.show()

accuracy = np.trace(cm_aligned) / np.sum(cm_aligned)
print("Clustering Accuracy:", accuracy)


# =====================================================
# 4️⃣ PCA VISUALIZATION
# =====================================================
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)
z_pca = pca.transform(z)

plt.figure()
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, s=20)
plt.scatter(z_pca[:, 0], z_pca[:, 1], c='red', marker='X', s=200)
plt.title("PCA – Colored by Cluster")
plt.show()

import numpy as np
import pandas as panda
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment


# ============================================================
# SCALER
# ============================================================
def scaler(x):
    x_scaled = np.zeros_like(x, dtype=float)
    for j in range(x.shape[1]):
        mean = np.mean(x[:, j])
        std = np.std(x[:, j])
        if std == 0:
            std = 1.0  # avoid divide by zero
        for i in range(x.shape[0]):
            x_scaled[i, j] = (x[i, j] - mean) / std
    return x_scaled


# ============================================================
# K-MEANS FUNCTIONS
# ============================================================
def generate(z, x):
    for i in range(z.shape[0]):
        z[i] = x[np.random.randint(0, x.shape[0])]
    return z


def assign(z, x):
    y = np.zeros(x.shape[0], dtype=int)
    for i in range(x.shape[0]):
        d = [np.linalg.norm(x[i] - z[k]) for k in range(z.shape[0])]
        y[i] = np.argmin(d)
    return y


def centroid(x, y, z_old):
    z_new = np.zeros_like(z_old)
    for k in range(z_old.shape[0]):
        pts = x[y == k]
        if len(pts) > 0:
            z_new[k] = np.mean(pts, axis=0)
        else:
            z_new[k] = x[np.random.randint(0, x.shape[0])]  # reinit empty cluster
    return z_new


def inertia(x, y, z):
    return sum(np.linalg.norm(x[i] - z[y[i]])**2 for i in range(len(x)))


# ============================================================
# LOAD DATA
# ============================================================
data = panda.read_csv(
    "/Users/satviksingh/Documents/manas_projects/classification/seeds_data.csv"
)

x = scaler(data.drop("Class", axis=1).values)
y_true = data["Class"].values


# ============================================================
# ELBOW + SILHOUETTE
# ============================================================
Ks = range(2, 8)
inertias, silhouettes = [], []

for K in Ks:
    z = generate(np.zeros((K, x.shape[1])), x)

    for _ in range(10):
        y = assign(z, x)
        z = centroid(x, y, z)

    inertias.append(inertia(x, y, z))

    if len(np.unique(y)) == K and len(np.unique(y)) > 1:
        silhouettes.append(silhouette_score(x, y))
    else:
        silhouettes.append(np.nan)


# ============================================================
# FINAL K = 3
# ============================================================
K = 3
z = generate(np.zeros((K, x.shape[1])), x)

loss_per_epoch = []
for _ in range(10):
    y = assign(z, x)
    z = centroid(x, y, z)
    loss_per_epoch.append(inertia(x, y, z))


# ============================================================
# CONFUSION MATRIX + ALIGNMENT
# ============================================================
cm = confusion_matrix(y_true, y)

row_ind, col_ind = linear_sum_assignment(-cm)
label_map = {col: row for row, col in zip(row_ind, col_ind)}

y_aligned = np.array([label_map[label] for label in y])
cm_aligned = confusion_matrix(y_true, y_aligned)

accuracy = np.trace(cm_aligned) / np.sum(cm_aligned)


# ============================================================
# PCA
# ============================================================
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)
z_pca = pca.transform(z)


# ============================================================
# STATS
# ============================================================
sizes = [np.sum(y == k) for k in range(K)]

means, stds = [], []
for k in range(K):
    means.append(np.mean(x[y == k], axis=0))
    stds.append(np.std(x[y == k], axis=0))

means = np.array(means)
stds = np.array(stds)


# ============================================================
# ALL GRAPHS IN ONE FIGURE
# ============================================================
fig = plt.figure(figsize=(18, 20))
gs = fig.add_gridspec(4, 3)

# Elbow
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(Ks, inertias, marker='o')
ax1.set_title("Elbow Curve")
ax1.set_xlabel("K")
ax1.set_ylabel("Inertia")

# Silhouette
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(Ks, silhouettes, marker='o')
ax2.set_title("Silhouette Score vs K")
ax2.set_xlabel("K")
ax2.set_ylabel("Score")

# Loss vs Epochs
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(range(len(loss_per_epoch)), loss_per_epoch, marker='o')
ax3.set_title("Loss (Inertia) vs Epochs")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Inertia")

# Confusion matrix (unaligned)
ax4 = fig.add_subplot(gs[1, 0])
im1 = ax4.imshow(cm)
ax4.set_title("Confusion Matrix (Unaligned)")
fig.colorbar(im1, ax=ax4)

# Confusion matrix (

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def graph(X, y, x_label, y_label, step = 4, color = 'viridis', path = True, fs = 14):
    plt.figure()
    plt.scatter(X, y, c=y, s=50, cmap=color)
    if path:
        plt.plot(X, y, color='tab:blue', alpha=0.5)
    plt.xlabel(x_label, fontsize=fs)
    plt.ylabel(y_label, fontsize=fs)
    plt.grid(True, alpha=0.5)
    plt.xticks(np.arange(min(X), max(X)+2, step))
    plt.show()

def get_distances_sum(X, labels, centroids):
    suma = 0
    for i, (x, label) in enumerate(zip(X, labels)):
        if label == -1:
            continue
        dist = np.linalg.norm(x - centroids[label])
        suma += dist
    return suma
        
def plot_ellipse_transformed(mean, cov, ax=None, scale = 1, color='black', alpha=0.5):
    if cov.shape != (2, 2):
        return
    vals, vecs = np.linalg.eig(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * scale * np.sqrt(vals)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                  edgecolor=color, fc='none', lw=1.3, alpha=alpha, zorder=3)

    ax.add_patch(ellipse)


def get_centroids(X, labels):
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    n_clusters = len(unique_labels)
    
    centroids = np.zeros((n_clusters, X.shape[1]))
    for i, label in enumerate(unique_labels):
        centroids[i] = np.mean(X[labels == label], axis=0)
    return centroids

def plot_dbscan_clusters(X, labels, fs = 14):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab20', s=10, alpha=0.6)
    
    ax.set_title(f'DBSCAN: {n_clusters} clusters found')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.grid()

def sub_plot(X, axs, axs_i, axs_j, labels, params, fs=14):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    cmap = plt.cm.tab20
    colors = []
    for label in labels:
        if label == -1:
            colors.append(np.array([0, 0, 0, 1]))
        else:
            colors.append(cmap(label % 20))
    
    colors = np.array(colors)
    
    axs[axs_i, axs_j].scatter(
        X[:, 0], X[:, 1], 
        c=colors,
        s=10, 
        alpha=0.4
    )
    
    axs[axs_i, axs_j].set_title(
        f'DBSCAN: {n_clusters} clusters found\neps: {params[0]}, min_samples: {params[1]}', 
        fontsize=fs
    )
    axs[axs_i, axs_j].set_xlim(0, 1)
    axs[axs_i, axs_j].set_ylim(0, 1)
    axs[axs_i, axs_j].grid()

def get_avas(X):
    cov = np.cov(X, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    return eigenvalues

import numpy as np
import matplotlib.pyplot as plt

def graficar_valores_singulares(X, components=100):
    fs = 14
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    
    S_comp = S[:components]

    plt.figure(figsize=(10, 5))
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(S_comp)) for i in range(len(S_comp))]
    plt.bar(range(1, len(S_comp) + 1), S_comp, color=colors)
    plt.ylabel('Valor singular', fontsize=fs)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
# def matriz_de_similaridad(X, sigma=1.0):
#     X_norm = np.sum(X ** 2, axis=1)
#     D = np.sqrt(X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T))
#     S = np.exp(-D**2 / (2 * sigma**2))
#     return S

import numpy as np
import matplotlib.pyplot as plt

def graph(X, y, x_label, y_label, path = True, fs = 14):
    plt.figure()
    #font size
    plt.scatter(X, y, c=y, s=50, cmap='viridis')
    if path:
        # conectamos los puntos
        plt.plot(X, y, color='black', alpha=0.5)
    plt.xlabel(x_label, fontsize=fs)
    plt.ylabel(y_label, fontsize=fs)
    plt.grid()
    plt.xticks(np.arange(min(X), max(X)+2, 4))
    plt.show()

# def get_distances_sum(X, labels, centroids):
#     distances = np.zeros((len(labels), len(centroids)))
#     suma = 0
#     for i, (x, label) in enumerate(zip(X, labels)):
#         dist = np.linalg.norm(x - centroids[label])
#         suma += dist
#     return suma

def get_distances_sum(X, labels, centroids):
    suma = 0
    for i, (x, label) in enumerate(zip(X, labels)):
        if label == -1:  # Skip noise points
            continue
        dist = np.linalg.norm(x - centroids[label])
        suma += dist
    return suma
        

import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

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
    unique_labels = unique_labels[unique_labels != -1]  # Exclude noise (-1)
    n_clusters = len(unique_labels)
    
    centroids = np.zeros((n_clusters, X.shape[1]))
    for i, label in enumerate(unique_labels):
        centroids[i] = np.mean(X[labels == label], axis=0)
    return centroids

def plot_dbscan_clusters(X, labels, core_samples=None, fs = 14):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    # Plot points with cluster colors
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab20', s=10, alpha=0.6)
    
    # Set title and add grid
    ax.set_title(f'DBSCAN: {n_clusters} clusters found')
    
    # Set plot limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.grid()
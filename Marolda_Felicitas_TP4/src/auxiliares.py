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

def get_distances_sum(X, labels, centroids):
    distances = np.zeros((len(labels), len(centroids)))
    suma = 0
    for i, (x, label) in enumerate(zip(X, labels)):
        dist = np.linalg.norm(x - centroids[label])
        suma += dist
    return suma
        

import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

def plot_ellipse_transformed(mean, cov, ax=None, color='black', alpha=0.5):
    if cov.shape != (2, 2):
        return
    vals, vecs = np.linalg.eig(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                  edgecolor=color, fc='none', lw=1.3, alpha=alpha, zorder=3)

    ax.add_patch(ellipse)

def plot_dbscan_clusters(X, labels, core_samples=None, fs = 14):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Plot
    plt.figure(figsize=(8, 6))
    
    # Plot noise points
    noise_mask = (labels == -1)
    plt.scatter(X[noise_mask, 0], X[noise_mask, 1], 
                c='k', marker='.', alpha=0.3,
                label='Noise')
    
    # Plot each cluster with a different color
    unique_labels = set(labels) - {-1}
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, (label, col) in enumerate(zip(unique_labels, colors)):
        cluster_mask = (labels == label)
        plt.scatter(X[cluster_mask, 0], X[cluster_mask, 1], 
                    c=[col], marker='o', alpha=0.7,
                    label=f'Cluster {label}')
        
    # Highlight core samples if provided
    if core_samples is not None and len(core_samples) > 0:
        plt.scatter(X[core_samples, 0], X[core_samples, 1], 
                    s=80, c='none', alpha=0.7, edgecolors='black',
                    linewidths=1, marker='o',
                    label='Core samples')
        
    plt.title(f'DBSCAN Clustering: {n_clusters} clusters found', fontsize=fs)
    plt.legend(fontsize=fs)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
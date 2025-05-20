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

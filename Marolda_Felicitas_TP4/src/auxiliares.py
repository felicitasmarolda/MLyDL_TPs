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
        
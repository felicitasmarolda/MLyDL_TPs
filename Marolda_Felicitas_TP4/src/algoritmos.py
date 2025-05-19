import numpy as np
import matplotlib.pyplot as plt

def k_means(X, k, max_iters=1000, threshold = 1e-8):
    n_samples, n_features = X.shape

    # inicializamos los centroides random
    idxs = np.random.choice(n_samples, k, replace=False)
    centroids = X[idxs]

    for _ in range(max_iters):
        # Asignamos cluster
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # nuevos centorides
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])

        # si converge..
        if np.linalg.norm(new_centroids - centroids) <= threshold:
            break
        centroids = new_centroids

    return centroids, labels

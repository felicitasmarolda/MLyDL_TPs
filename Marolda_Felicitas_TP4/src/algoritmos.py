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

def GMM(X, k, centroids_init=None, max_iters=300, threshold=1e-4):
    n_samples, n_features = X.shape
    
    # Inicializamos
    if centroids_init is None:
        # inicializamos random
        idxs = np.random.choice(n_samples, k, replace=False)
        medias = X[idxs]
    else:
        medias = centroids_init
    
    # covarianzas como matriz identidad
    covariances = np.array([np.cov(X.T) + np.eye(n_features) * 1e-6 for _ in range(k)])
    
    # pesos uniformes
    weights = np.ones(k) / k
    
    # inicializamos responsabilidades
    responsibilities = np.zeros((n_samples, k))
    
    log_likelihood_history = []
    medias_history = []
    covariances_history = []
    weights_history = []


    prev_log_likelihood = -np.inf
    
    for iteration in range(max_iters):
        # print("iteration", iteration)
        # E-step: calcular responsabilidades
        for n in range(n_samples):
            for j in range(k):
                responsibilities[n, j] = weights[j] * multivariate_gaussian_pdf(X[n], medias[j], covariances[j])
            responsibilities[n] /= np.sum(responsibilities[n])
        # print("responsibilities", responsibilities)
        
        # M-step: actualizar parámetros
        for j in range(k):
            N_j = np.sum(responsibilities[:, j])
            medias[j] = np.sum(responsibilities[:, j][:, np.newaxis] * X, axis=0) / N_j
            covariances[j] = np.dot((responsibilities[:, j][:, np.newaxis] * (X - medias[j])).T, (X - medias[j])) / N_j
            covariances[j] += np.eye(n_features) * 1e-6  # Regularización para evitar singularidad
            weights[j] = N_j / n_samples   
        
        # log likelihood
        log_likelihood = 0
        for n in range(n_samples):
            log_likelihood += np.log(np.sum([weights[j] * multivariate_gaussian_pdf(X[n], medias[j], covariances[j]) for j in range(k)]))

        log_likelihood_history.append(log_likelihood)
        medias_history.append(medias.copy())
        covariances_history.append(covariances.copy())
        weights_history.append(weights.copy())


        # si converge..
        if np.abs(log_likelihood - prev_log_likelihood) < threshold:
            break
        # usamos allclose para convergencia
        if iteration > 2 and np.allclose(medias, medias_history[-1], rtol = threshold, atol=threshold) and np.allclose(covariances, covariances_history[-1], rtol = threshold, atol=threshold) and np.allclose(weights, weights_history[-1], rtol = threshold, atol=threshold):
            break

        prev_log_likelihood = log_likelihood
    
    # labels
    # print("responsibilities", responsibilities)
    labels = np.argmax(responsibilities, axis=1)

    return medias, covariances, weights, responsibilities, log_likelihood_history, labels


def multivariate_gaussian_pdf(x, media, cov):
    d = len(x)
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    norm_const = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(cov_det))
    x_diff = x - media
    exponent = -0.5 * np.dot(x_diff, np.dot(cov_inv, x_diff))
    output = norm_const * np.exp(exponent)
    # print("output", output)
    return output

# def DBSCAN(X, eps, k):
#     # k = min samples
#     n_samples = X.shape[0]
    
#     # Initialize labels to unvisited (-2)
#     labels = np.full(n_samples, -2)
    
#     # Find neighbors for each point
#     neighbors = []
#     for i in range(n_samples):
#         dists = np.linalg.norm(X - X[i], axis=1)
#         neighbors.append(np.where(dists <= eps)[0])

#     # identify core points
#     to_join_points = np.array([i for i, neighbor_indices in enumerate(neighbors) if len(neighbor_indices) >= k])
    
#     # definimos ruido como los puntos que no tienen suficientes vecinos
#     for i in range(n_samples):
#         if i not in to_join_points:
#             labels[i] = -1
    
#     # clusterizamos
#     cluster_id = 0
#     for i in range(n_samples):
#         # Skip if already visited or not a core point
#         if labels[i] != -2 or i not in to_join_points:
#             continue
        
#         # Start a new cluster
#         labels[i] = cluster_id
        
#         # Process neighbors (BFS approach)
#         queue = list(neighbors[i])
#         while queue:
#             neighbor = queue.pop(0)
            
#             # If neighbor is unvisited or noise
#             if labels[neighbor] < 0:  # Unvisited (-2) or noise (-1)
#                 # Add to cluster
#                 labels[neighbor] = cluster_id
                
#                 # If it's a core point, add its neighbors to the queue
#                 if neighbor in to_join_points:
#                     queue.extend([n for n in neighbors[neighbor] if labels[n] < 0])
        
#         # Move to next cluster
#         cluster_id += 1
    
#     return labels, to_join_points

def DBSCAN(X, eps, k):
    n_samples = X.shape[0]
    labels = np.full(n_samples, -2)  # -2: unvisited, -1: noise, >=0: cluster ID
    neighbors = [np.where(np.linalg.norm(X - X[i], axis=1) <= eps)[0] for i in range(n_samples)]
    to_join_points = np.array([i for i, neigh in enumerate(neighbors) if len(neigh) >= k])
    
    # Label noise points
    labels[np.isin(np.arange(n_samples), to_join_points, invert=True)] = -1

    cluster_id = 0
    for i in range(n_samples):
        if labels[i] != -2 or i not in to_join_points:
            continue
        
        # Start BFS for cluster expansion
        queue = [i]
        labels[i] = cluster_id
        
        while queue:
            current = queue.pop(0)
            
            # Process only unvisited/noise neighbors (skip already clustered)
            for neighbor in neighbors[current]:
                if labels[neighbor] == -2 or labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                    if neighbor in to_join_points:
                        queue.append(neighbor)
        
        cluster_id += 1
    
    return labels, to_join_points
import numpy as np

def min_max(X):
    # pone entre 0 y 1 usando el minima y maximo de cada columna
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X = (X - X_min) / (X_max - X_min)
    return X
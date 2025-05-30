import numpy as np

def min_max(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X = (X - X_min) / (X_max - X_min)
    return X
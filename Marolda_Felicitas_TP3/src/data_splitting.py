import numpy as np

def split_data(X, y, test_size = 0.2, val_size = 0.2, random_state = 42):
    np.random.seed(random_state)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_size = int(len(X) * test_size)
    val_size = int(len(X) * val_size)
    
    X_test = X[indices[:test_size]]
    y_test = y[indices[:test_size]]
    
    X_val = X[indices[test_size:test_size + val_size]]
    y_val = y[indices[test_size:test_size + val_size]]
    
    X_train = X[indices[test_size + val_size:]]
    y_train = y[indices[test_size + val_size:]]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize(X, number = 255):
    # dividimos todo por 255
    return X / number

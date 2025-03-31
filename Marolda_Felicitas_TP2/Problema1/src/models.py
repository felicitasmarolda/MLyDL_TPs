import numpy as np

class Logistic_Regression:
    def __init__(self, X, y, threshold=0.01, max_iterations=1000, L2 = 0):
        """
        threshold: threshold value to classify as class 1 (default 0.5)
        max_iter: max number of iterations for gradient descent
        learning_rate: learning rate for gradient descent
        """
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.y = y
        self.threshold = threshold
        self.max_iter = max_iterations
        self.learning_rate = L2
        self.coef = np.zeros(X.shape[1])

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


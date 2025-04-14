import numpy as np
class Logistic_Regression:
    def __init__(self, X, y, features, L2 = 0, threshold=0.5, max_iterations=1000, fit = True):
        """
        threshold: threshold value to classify as class 1 (default 0.5)
        max_iter: max number of iterations for gradient descent
        learning_rate: learning rate for gradient descent
        """
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.y = np.array(y)
        self.features = features
        self.threshold = threshold
        self.max_iter = max_iterations
        self.L2 = L2
        self.learning_rate = 0.01
        self.coef = np.zeros(self.X.shape[1])
        self.coef_trace = []
        if fit:
            self.fit()

    def fit(self):
        self.gradient_descent()
    
    def _sigmoid(self, z):
        z = np.array(z)
        return 1 / (1 + np.exp(-z))


    def loss_function(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def gradient(self, y_pred):
        y_pred = y_pred.reshape(-1) 
        self.y = self.y.reshape(-1) 
        return (self.X.T @ (y_pred - self.y)) / self.y.size + self.L2 * self.coef
        

    def gradient_descent(self):
        for _ in range(self.max_iter):
            z = np.dot(self.X, self.coef)
            y_hat = self._sigmoid(z)
            gradient = self.gradient(y_hat)
            self.coef -= self.learning_rate * gradient
            self.coef_trace.append(self.coef.copy())
    
    def predict_proba(self, X):
        X = np.column_stack((np.ones(X.shape[0]), X))
        y_pred = self._sigmoid(np.dot(X, self.coef))
        return y_pred
    
    def predict(self, X):
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba >= self.threshold).astype(int)
    

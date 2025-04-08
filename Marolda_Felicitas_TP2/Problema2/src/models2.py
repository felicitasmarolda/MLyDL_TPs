import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Logistic_Regression_Multiclass:
    def __init__(self, X, y, features, L2=0, threshold = 0.5, max_iterations=1000, fit=True):
        """
        threshold: threshold value to classify as class 1 (default 0.5)
        max_iter: max number of iterations for gradient descent
        learning_rate: learning rate for gradient descent
        """
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.y = np.array(y).flatten()
        # print("y:", self.y)
        self.classes = np.unique(self.y)
        self.y = pd.Categorical(self.y, categories=self.classes).codes
        self.features = features
        self.threshold = threshold
        self.max_iter = max_iterations
        self.L2 = L2
        self.learning_rate = 0.01
        self.n_classes = len(self.classes)
        self.coef = np.zeros((self.n_classes, self.X.shape[1]))
        # self.coef = np.zeros((len(np.unique(y)), self.X.shape[1]))
        self.coef_trace = []
        # print("y valores únicos:", np.unique(self.y))
        # print("shape de coef:", self.coef.shape)


        if fit:
            self.fit()
        
    def fit(self):
        self.gradient_descent()

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def loss_function(self, y, y_pred):
        return -np.mean(np.sum(y * np.log(y_pred + 1e-10), axis=1)) + self.L2 * np.sum(self.coef ** 2)
    
    def gradient(self, y_pred):
        y_pred = y_pred.reshape(-1, self.coef.shape[0])
        # self.y = self.y.reshape(-1, 1)
        y_one_hot = np.zeros((self.y.size, self.coef.shape[0]))
        # print("y_one_hot shape:", y_one_hot.shape)
        # print("y_pred shape:", y_pred.shape)
        # print("self.y shape:", self.y.shape)
        y_one_hot[np.arange(self.y.size), self.y.flatten()] = 1
        return (self.X.T @ (y_pred - y_one_hot)) / self.y.size + self.L2 * self.coef.T
    

    def gradient_descent(self):
        for _ in range(self.max_iter):
            z = np.dot(self.X, self.coef.T)
            y_hat = self.softmax(z)
            gradient = self.gradient(y_hat)
            self.coef -= self.learning_rate * gradient.T
            self.coef_trace.append(self.coef.copy())
            if np.linalg.norm(gradient) < 1e-6:
                break

    def predict_proba(self, X):
        X = np.column_stack((np.ones(X.shape[0]), X))
        z = np.dot(X, self.coef.T)
        return self.softmax(z)
    
    def predict(self, X):
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)
import numpy as np
class KNNClassifier:
    def __init__(self, X, y, features, k=3):
        self.X = np.array(X)
        self.y = np.array(y)
        self.features = features
        self.k = k

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            distances = [self.euclidean_distance(x_test, x_train) for x_train in self.X]    # Calculate distances to all training points
            k_indices = np.argsort(distances)[:self.k]  # Get indices of k nearest neighbors
            k_nearest_labels = [self.y[i] for i in k_indices] # Get labels of k nearest neighbors
            most_common = np.bincount(k_nearest_labels).argmax()    # Find the most common label among the k nearest neighbors
            predictions.append(most_common)   # Append the prediction for the current test point    
        return np.array(predictions)
    
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
        print(f"Coef shape: {self.coef.shape}")
        self.coef_trace = []
        if fit:
            self.fit()

    def fit(self):
        self.gradient_descent()
    
    def coef_names(self):
        """Imprime los coeficientes con los nombres de sus respectivas variables de forma prolija en 
        una tabla."""
        table = "\n" + "{:<20}".format("| Variable")
        for feature in self.features:
            table += " | {:>15}".format(feature)
        table += "|"
        table += "\n" + "{:<20}".format("| Coeficiente")
        for coef in self.coef:
            table += " | {:>15.6f}".format(coef[0])
        table += "|"
        return table
    
    def _sigmoid(self, z):
        z = np.array(z)  # Ensure z is a NumPy array
        # print(f"Type of z: {type(z)}, Value of z: {z}")
        return 1 / (1 + np.exp(-z))


    def loss_function(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    
    def gradient(self, y_pred):
        y_pred = y_pred.reshape(-1)  # Asegura que y_pred sea un vector plano
        self.y = self.y.reshape(-1)  # Asegura que self.y sea un vector plano
        # print(self.y.shape)
        # print((y_pred - self.y).shape)
        return (self.X.T @ (y_pred - self.y)) / self.y.size + self.L2 * self.coef
        # return np.dot(self.X.T, (y_pred - self.y)) / self.y.size + self.learning_rate * self.coef
        

    def gradient_descent(self):
        for _ in range(self.max_iter):
            # print("X: ", self.X)
            # print("Coef: ", self.coef)
            z = np.dot(self.X, self.coef)
            y_hat = self._sigmoid(z)
            # print("Y_hat: ", y_hat.shape)
            gradient = self.gradient(y_hat)
            self.coef -= self.learning_rate * gradient
            self.coef_trace.append(self.coef.copy())
        # print(self.coef)
        
    def predict(self, X):
        X = np.column_stack((np.ones(X.shape[0]), X))
        y_pred = self._sigmoid(np.dot(X, self.coef))
        # print(f"Predictions prob: {y_pred}")
        return (y_pred >= self.threshold).astype(int)
    

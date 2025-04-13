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
        class_indices = np.argmax(y_proba, axis=1)
        return self.classes[class_indices]

# import numpy as np

# class DecisionTree:
#     def __init__(self, X, labels, features, max_depth=10, max_information_gain=0.95, features_perc=0.6):
#         self.X = X
#         self.labels = labels
#         self.features = features  # estos son índices
#         self.max_depth = max_depth
#         self.information_gain = max_information_gain
#         self.features_perc = features_perc
#         self.n_classes = len(np.unique(labels))
#         self.tree = None
#         self.fit = True
#         if self.fit:
#             self.tree = self.build_tree(X, labels, features)

#     def get_best_division(self, X, y, features):
#         n_features = int(len(features) * self.features_perc)
#         selected_features = np.random.choice(features, n_features, replace=False)

#         best_feature = None
#         best_threshold = None
#         best_gain = -np.inf
#         best_left_indices = None
#         best_right_indices = None

#         for feature in selected_features:
#             thresholds = np.unique(X[:, feature])
#             for threshold in thresholds:
#                 left_indices = np.where(X[:, feature] <= threshold)[0]
#                 right_indices = np.where(X[:, feature] > threshold)[0]

#                 if len(left_indices) == 0 or len(right_indices) == 0:
#                     continue

#                 gain = self.information_gain_function(y, left_indices, right_indices)

#                 if gain > best_gain:
#                     best_gain = gain
#                     best_feature = feature
#                     best_threshold = threshold
#                     best_left_indices = left_indices
#                     best_right_indices = right_indices

#         return best_feature, best_threshold, best_left_indices, best_right_indices

#     def information_gain_function(self, y, left_indices, right_indices):
#         parent_entropy = self.entropy(y)
#         n = len(y)
#         n_left = len(left_indices)
#         n_right = len(right_indices)

#         if n_left == 0 or n_right == 0:
#             return 0

#         child_entropy = (n_left / n) * self.entropy(y[left_indices]) + (n_right / n) * self.entropy(y[right_indices])
#         return parent_entropy - child_entropy

#     def entropy(self, y):
#         if len(y) == 0:
#             return 0
#         p = np.bincount(y) / len(y)
#         return -np.sum(p * np.log2(p + 1e-10))

#     def build_tree(self, X, y, features, depth=0):
#         if len(np.unique(y)) == 1 or depth >= self.max_depth:
#             return np.bincount(y).argmax()

#         best_feature, best_threshold, left_indices, right_indices = self.get_best_division(X, y, features)

#         if best_feature is None:
#             return np.bincount(y).argmax()

#         left_tree = self.build_tree(X[left_indices], y[left_indices], features, depth + 1)
#         right_tree = self.build_tree(X[right_indices], y[right_indices], features, depth + 1)

#         return (best_feature, best_threshold, left_tree, right_tree)

#     def predict(self, X):
#         predictions = np.zeros(X.shape[0], dtype=int)
#         for i in range(X.shape[0]):
#             predictions[i] = self.traverse_tree(X[i], self.tree)
#         return predictions

#     def traverse_tree(self, x, tree):
#         if not isinstance(tree, tuple):
#             return tree

#         feature, threshold, left_tree, right_tree = tree

#         if x[feature] <= threshold:
#             return self.traverse_tree(x, left_tree)
#         else:
#             return self.traverse_tree(x, right_tree)


# class RandomForest:
#     def __init__(self, X, y, features, n_trees=10, max_depth=5, features_perc=0.6, data_perc=1):
#         self.X = X
#         self.y = np.array(y).flatten()
#         self.features = features
#         self.n_trees = n_trees
#         self.max_depth = max_depth
#         self.features_perc = features_perc
#         self.data_perc = data_perc
#         self.classes = np.unique(self.y)
#         self.fit = True
#         self.trees = []
#         if self.fit:
#             self.fit_()

#     def fit_(self):
#         for _ in range(self.n_trees):
#             n_samples = int(len(self.X) * self.data_perc)
#             indices = np.random.choice(range(len(self.X)), n_samples, replace=True)
#             X_sample = self.X[indices]
#             y_sample = self.y[indices]

#             tree = DecisionTree(
#                 X_sample,
#                 y_sample,
#                 self.features,
#                 max_depth=self.max_depth,
#                 features_perc=self.features_perc
#             )
#             self.trees.append(tree)

#     def predict_proba(self, X):
#         n_samples = X.shape[0]
#         n_classes = len(self.classes)
#         proba = np.zeros((n_samples, n_classes))

#         for tree in self.trees:
#             preds = tree.predict(X)
#             for i in range(n_samples):
#                 class_index = np.where(self.classes == preds[i])[0][0]
#                 proba[i, class_index] += 1

#         proba /= self.n_trees
#         return proba

#     def predict(self, X):
#         predictions = np.zeros((X.shape[0], self.n_trees), dtype=int)
#         for i, tree in enumerate(self.trees):
#             predictions[:, i] = tree.predict(X)

#         final_predictions = np.array([np.bincount(predictions[i]).argmax() for i in range(X.shape[0])])
#         return final_predictions

    
class DecisionTree:
    def __init__(self, X, labels, features, features_perc, max_depth=10, max_information_gain = 0.95):
        self.X = X
        self.labels = labels
        self.features = features
        self.max_depth = max_depth
        self.features_perc = features_perc
        self.information_gain = max_information_gain
        self.n_classes = len(np.unique(labels))
        self.tree = None
        self.fit = True
        if self.fit:
            self.tree = self.build_tree(X, labels, features)

    def get_best_division(self, X, y, features):
        best_feature = None
        best_threshold = None
        best_gain = -np.inf
        best_left_indices = None
        best_right_indices = None

        # n_features = int(len(features) * self.features_perc)
        # selected_features = np.random.choice(features, n_features, replace=False)
        n_features = int(len(self.features) * self.features_perc)
        selected_features = np.random.choice(range(len(self.features)), n_features, replace=False)

        for feature in selected_features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature] <= threshold)[0]
                right_indices = np.where(X[:, feature] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                gain = self.information_gain_function(y, left_indices, right_indices)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        return best_feature, best_threshold, best_left_indices, best_right_indices
    
    def information_gain_function(self, y, left_indices, right_indices):
        parent_entropy = self.entropy(y)
        n = len(y)
        n_left = len(left_indices)
        n_right = len(right_indices)

        if n_left == 0 or n_right == 0:
            return 0

        child_entropy = (n_left / n) * self.entropy(y[left_indices]) + (n_right / n) * self.entropy(y[right_indices])
        return parent_entropy - child_entropy
    
    def entropy(self, y):
        if len(y) == 0:
            return 0
        # print("y:", y) 
        p = np.bincount(y) / len(y)
        return -np.sum(p * np.log2(p + 1e-10))
    
    def build_tree(self, X, y, features, depth=0):
        if len(np.unique(y)) == 1 or depth >= self.max_depth:
            return np.bincount(y).argmax()

        best_feature, best_threshold, left_indices, right_indices = self.get_best_division(X, y, features)

        if best_feature is None:
            return np.bincount(y).argmax()

        left_tree = self.build_tree(X[left_indices], y[left_indices], features, depth + 1)
        right_tree = self.build_tree(X[right_indices], y[right_indices], features, depth + 1)

        return (best_feature, best_threshold, left_tree, right_tree)

    def predict(self, X):
        predictions = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            predictions[i] = self.traverse_tree(X[i], self.tree)
        return predictions

    def traverse_tree(self, x, tree):
        if not isinstance(tree, tuple):
            return tree

        feature, threshold, left_tree, right_tree = tree

        if x[feature] <= threshold:
            return self.traverse_tree(x, left_tree)
        else:
            return self.traverse_tree(x, right_tree)

class RandomForest:
    def __init__(self, X, y, features, n_trees=10, max_depth=5, features_perc = 0.6, data_perc = 1):
        self.X = X
        self.y = np.array(y).flatten()
        self.features = features
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.features_perc = features_perc
        self.data_perc = data_perc
        self.classes = np.unique(self.y)
        self.fit = True
        self.trees = []
        if self.fit:
            self.fit_()

    def fit_(self):
        for _ in range(self.n_trees):
            # sampleamos los datos para obetner un nevo dataset con data_perc * cantidad de datos como cantidad de datos
            n_samples = int(len(self.X) * self.data_perc)
            indices = np.random.choice(range(len(self.X)), n_samples, replace=True)
            X_sample = self.X[indices]
            y_sample = self.y[indices]

            tree = DecisionTree(X_sample, y_sample, self.features, self.features_perc, max_depth=self.max_depth)
            self.trees.append(tree)
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        proba = np.zeros((n_samples, n_classes))

        for tree in self.trees:
            preds = tree.predict(X)
            for i in range(n_samples):
                class_index = np.where(self.classes == preds[i])[0][0]
                proba[i, class_index] += 1

        proba /= self.n_trees
        return proba

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_trees), dtype=int)
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        
        # Voting
        final_predictions = np.array([np.bincount(predictions[i]).argmax() for i in range(X.shape[0])])
        return final_predictions
    
class LDAClassifier:
    def __init__(self, X, y, features, fit=True):
        self.X = X
        self.y = np.array(y).flatten()
        self.features = features
        self.classes = np.unique(self.y)
        print("self.classes:", self.classes)
        self.means = None
        self.cov_matrix = None
        self.priors = None
        if fit:
            self.fit_()

    def fit_(self):
        self.calculate_means()
        self.calculate_covariance_matrix()
        self.calculate_priors()

    def calculate_means(self):
        self.means = np.array([np.mean(self.X[self.y == c], axis=0) for c in self.classes])

    def calculate_covariance_matrix(self):
        n_samples, n_features = self.X.shape
        self.cov_matrix = np.zeros((n_features, n_features))
        
        for c in self.classes:
            class_samples = self.X[self.y == c]
            class_mean = np.mean(class_samples, axis=0)
            centered_samples = class_samples - class_mean
            self.cov_matrix += np.dot(centered_samples.T, centered_samples)
        
        # Normalize by the number of samples minus the number of classes
        self.cov_matrix /= (n_samples - len(self.classes))

    def calculate_priors(self):
        n_samples = len(self.y)
        self.priors = np.array([np.sum(self.y == c) / n_samples for c in self.classes])

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probabilities = np.zeros((n_samples, n_classes))

        for i, c in enumerate(self.classes):
            mean_diff = X - self.means[i]
            inv_cov_matrix = np.linalg.inv(self.cov_matrix)
            exponent = -0.5 * np.sum(np.dot(mean_diff, inv_cov_matrix) * mean_diff, axis=1)
            probabilities[:, i] = self.priors[i] * np.exp(exponent)

        # Normalize probabilities
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return probabilities
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes[class_indices]

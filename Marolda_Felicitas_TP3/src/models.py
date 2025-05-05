import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, X, y, X_val, y_val, activation_functions: list, nodes_in_layer: list, learning_rate: float = 0.1, epochs = 1000, weights_inicialies = 'He'):
        self.X = X
        self.y = np.eye(np.max(y) + 1)[y]
        self.y_val = np.eye(np.max(y) + 1)[y_val]
        self.activation_functions = activation_functions
        self.nodes_in_layer = [X.shape[1]] + nodes_in_layer + [self.y.shape[1]]
        self.layers = len(self.nodes_in_layer)
        self.learning_rate = learning_rate
        self.weights_inicialies = weights_inicialies
        self.a = {}
        self.a[0] = self.X
        self.z = {}
        self.delta = {}
        self.epochs = epochs
        self.losses = []
        self.losses_val = []
        self.gradients_weights = {}
        self.gradients_biases = {}
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.initialize_weights()

        # fit
        self.fit(X_val, y_val)

    def initialize_weights(self):
        if self.weights_inicialies == 'He':
            for i in range(self.layers - 1):
                weight = np.random.randn(self.nodes_in_layer[i], self.nodes_in_layer[i + 1]) * np.sqrt(2 / self.nodes_in_layer[i])
                bias = np.zeros((1, self.nodes_in_layer[i + 1]))
                self.weights.append(weight)
                self.biases.append(bias)
        else:
            for i in range(self.layers - 1):
                weight = np.random.randn(self.nodes_in_layer[i], self.nodes_in_layer[i + 1])
                bias = np.zeros((1, self.nodes_in_layer[i + 1]))
                self.weights.append(weight)
                self.biases.append(bias)

    def softmax(self, z)-> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def ReLU(self, z)-> np.ndarray:
        return np.maximum(0, z)
    
    def ReLU_derivative(self, z)-> np.ndarray:
        return np.where(z > 0, 1, 0)
    
    def cross_entropy_loss(self, y_pred, y = None):
        if y is None:
            y = self.y
        else:
            y = np.eye(np.max(y) + 1)[y]
        m = self.X.shape[0]
        loss = -np.sum(y * np.log(y_pred + 1e-8)) / m
        return loss
    def cross_entropy_loss_derivative(self, y_pred, y_true):
        return (y_pred - y_true)

    def forward_pass(self, a):
        for i in range(self.layers - 1):
            self.z[i] = np.dot(a, self.weights[i]) + self.biases[i]
            if self.activation_functions[i] == 'ReLU':
                a = self.ReLU(self.z[i])
            elif self.activation_functions[i] == 'softmax':
                a = self.softmax(self.z[i])
            self.a[i+1] = a
        return a
    
    def backward_pass(self, y_pred) -> None:
        m = self.X.shape[0]
        L = self.layers - 1
        self.delta[L] = self.cross_entropy_loss_derivative(y_pred, self.y)

        for i in reversed(range(L)):
            self.gradients_weights[i] = np.dot(self.a[i].T, self.delta[i + 1]) / m
            self.gradients_biases[i] = np.sum(self.delta[i + 1], axis=0, keepdims=True) / m

            if i != 0:
                if self.activation_functions[i-1] == 'ReLU':
                    dz = np.dot(self.delta[i + 1], self.weights[i].T)
                    self.delta[i] =dz * self.ReLU_derivative(self.z[i-1])              
                else:
                    raise ValueError("Unsupported activation function for backpropagation")
        
    def gradient_descent(self) -> None:
        for i in range(self.layers - 1):
            self.weights[i] -= self.learning_rate * self.gradients_weights[i]
            self.biases[i] -= self.learning_rate * self.gradients_biases[i]

    def backpropagation(self) -> None:
        y_pred = self.forward_pass(self.a[0])
        self.backward_pass(y_pred)

    def fit(self, X_val, y_val) -> None:
        for epoch in range(self.epochs):
            self.backpropagation()
            self.gradient_descent()
            y_pred = self.forward_pass(self.X)
            loss = self.cross_entropy_loss(y_pred)
            self.losses.append(loss)
            
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward_pass(X_val)
                val_loss = self.cross_entropy_loss(y_val_pred, y_val)
                self.losses_val.append(val_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
                print(f"loss val: {val_loss}")
            
        # graph the losses
        self.graph_losses()

    def graph_losses(self):
        plt.plot(self.losses, label='Train Loss')
        if self.losses_val:
            plt.plot(self.losses_val, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.legend()
        plt.show()



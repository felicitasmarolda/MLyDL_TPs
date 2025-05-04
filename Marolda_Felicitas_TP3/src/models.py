import numpy as np

class NeuralNetwork:
    def __init__(self, X, y, layers:int, activation_functions: list, nodes_in_layer: list, learning_rate: float = 0.01, epochs = 1000, weights_inicialies = 'He'):
        self.X = X
        self.y = y
        self.layers = layers
        self.activation_functions = activation_functions
        self.nodes_in_layer = nodes_in_layer
        self.learning_rate = learning_rate
        self.weights_inicialies = weights_inicialies
        self.a = {}
        self.a[0] = self.X
        self.z = {}
        self.delta = {}
        self.epochs = epochs
        self.losses = []
        self.gradients_weights = {}
        self.gradients_biases = {}
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.initialize_weights()

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
    
    def cross_entropy_loss(self, y_pred):
        m = self.X.shape[0]
        loss = -np.sum(self.y * np.log(y_pred + 1e-8)) / m
        return loss
    
    def cross_entropy_loss_derivative(self, y_pred):
        return (y_pred - self.y) / self.y.shape[0]

    def forward_pass(self, a) -> np.ndarray:
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
        self.delta[L] = self.cross_entropy_loss_derivative(y_pred)

        for i in range(L - 1, 0, -1):
            if self.activation_functions[i] == 'ReLU':
                self.delta[i] = np.dot(self.delta[i + 1], self.weights[i].T) * self.ReLU_derivative(self.z[i])
            else:
                raise ValueError("Unsupported activation function for backpropagation")
        
        for i in range(L):
            self.gradients_weights[f'W{i}'] = np.dot(self.a[i].T, self.delta[i + 1]) / m
            self.gradients_biases[f'b{i}'] = np.sum(self.delta[i + 1], axis=0, keepdims=True) / m
            self.weights[i] -= self.learning_rate * self.gradients_weights[f'W{i}']
            self.biases[i] -= self.learning_rate * self.gradients_biases[f'b{i}']


    def backpropagation(self) -> None:
        y_pred = self.forward_pass(self.a[0])
        self.backward_pass(y_pred)

    def train()


    

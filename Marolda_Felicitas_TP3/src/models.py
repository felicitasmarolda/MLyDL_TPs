import numpy as np

class NeuralNetwork:
    def __init__(self, X, y, layers:int, activation_functions: list, nodes_in_layer: list, learning_rate: float = 0.01, epochs = 1000, weights_inicialies = 'He'):
        self.X = X
        self.y = np.eye(np.max(y) + 1)[y]
        self.layers = layers + 2
        self.activation_functions = activation_functions
        self.nodes_in_layer = [X.shape[1]] + nodes_in_layer + [self.y.shape[1]]
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

        # fit
        self.fit()

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

        for i in reversed(range(L)):
            self.gradients_weights[i] = np.dot(self.a[i].T, self.delta[i + 1]) / m
            self.gradients_biases[i] = np.sum(self.delta[i + 1], axis=0, keepdims=True) / m
            
            # self.weights[i] -= self.learning_rate * self.gradients_weights[i]
            # self.biases[i] -= self.learning_rate * self.gradients_biases[i]
            # PONER EN LA FUNCIÓN DE GRADIENTE DESCENDIENTE
            
            if i != 0:
                if self.activation_functions[i-1] == 'ReLU':
                    dz = np.dot(self.delta[i + 1], self.weights[i].T)
                    self.delta[i] =dz * self.ReLU_derivative(self.z[i-1])              
                else:
                    raise ValueError("Unsupported activation function for backpropagation")
        

    def backpropagation(self) -> None:
        y_pred = self.forward_pass(self.a[0])
        self.backward_pass(y_pred)

    def fit(self) -> None:
        for epoch in range(self.epochs):
            self.backpropagation()
            if epoch % 100 == 0:
                loss = self.cross_entropy_loss(self.a[self.layers - 1])
                self.losses.append(loss)
                print(f'Epoch {epoch}, Loss: {loss}')



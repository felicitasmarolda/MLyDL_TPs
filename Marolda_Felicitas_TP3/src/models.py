import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, X, y, X_val, y_val, activation_functions: list, nodes_in_layer: list, mejora = None, learning_rate: float = 0.1, epochs = 1000, weights_inicialies = 'He'):
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

        # mejoras
        if mejora == None:
            self.mejora = {}
        else:
            self.mejora = mejora

        if self.mejora.get("ADAM", False):
            # Inicialización para Adam
            self.beta1 = 0.9  # Factor de decaimiento para el primer momento
            self.beta2 = 0.999  # Factor de decaimiento para el segundo momento
            self.epsilon = 1e-8
            self.m_t_weights = [np.zeros_like(w) for w in self.weights]
            self.v_t_weights = [np.zeros_like(w) for w in self.weights]
            self.m_t_biases = [np.zeros_like(b) for b in self.biases]
            self.v_t_biases = [np.zeros_like(b) for b in self.biases]
            self.t = 0

        # fit
        if self.mejora.get("Mini batch stochastic gradient descent", False):
            self.fit_mini_batch(X_val, y_val)
        else:
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
    
    def backward_pass(self, y_pred, y, X) -> None:
        m = X.shape[0]
        L = self.layers - 1
        self.delta[L] = self.cross_entropy_loss_derivative(y_pred, y)

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

    def backpropagation(self, y = None, X = None) -> None:
        if y is None:
            y = self.y
        if X is None:
            X = self.X
        self.a[0] = X

        y_pred = self.forward_pass(self.a[0])
        self.backward_pass(y_pred, y, X)

    def fit(self, X_val, y_val) -> None:
        for epoch in range(self.epochs):
            self.backpropagation()

            # gradient descent mejora
            if self.mejora.get("Rate scheduling lineal", False):
                self.gradient_descent_rate_scheduling_lineal(epoch)
            elif self.mejora.get("ADAM", False):
                self.gradient_descent_adam()
            else:
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


    # mejoras
    def gradient_descent_rate_scheduling_lineal(self, epoch):
        lr_init = self.learning_rate
        lr_min = 0.001  # tasa mínima
        decay_ratio = epoch / self.epochs
        current_lr = max(lr_init * (1 - decay_ratio), lr_min)

        for i in range(self.layers - 1):
            self.weights[i] -= current_lr * self.gradients_weights[i]
            self.biases[i] -= current_lr * self.gradients_biases[i]

    def gradient_descent_adam(self):
        self.t += 1  # Incrementamos el contador de pasos
        
        for i in range(self.layers - 1):
            # --- Actualización para pesos ---
            # 1. Calcular momentos (media y varianza)
            self.m_t_weights[i] = self.beta1 * self.m_t_weights[i] + (1 - self.beta1) * self.gradients_weights[i]
            self.v_t_weights[i] = self.beta2 * self.v_t_weights[i] + (1 - self.beta2) * (self.gradients_weights[i]**2)
            
            # 2. Corrección de bias (para contrarrestar la inicialización en 0)
            m_hat_w = self.m_t_weights[i] / (1 - self.beta1**self.t)
            v_hat_w = self.v_t_weights[i] / (1 - self.beta2**self.t)
            
            # 3. Actualización de pesos con tasa adaptativa
            self.weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            
            # --- Actualización para biases ---
            # 1. Calcular momentos (media y varianza)
            self.m_t_biases[i] = self.beta1 * self.m_t_biases[i] + (1 - self.beta1) * self.gradients_biases[i]
            self.v_t_biases[i] = self.beta2 * self.v_t_biases[i] + (1 - self.beta2) * (self.gradients_biases[i]**2)
            
            # 2. Corrección de bias
            m_hat_b = self.m_t_biases[i] / (1 - self.beta1**self.t)
            v_hat_b = self.v_t_biases[i] / (1 - self.beta2**self.t)
            
            # 3. Actualización de biases con tasa adaptativa
            self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def fit_mini_batch(self, X_val, y_val, batch_size = 32) -> None:
        for epoch in range(self.epochs):
            
            # mezcla los datos
            indices = np.arange(self.X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]

            # dividir en batches
            for i in range(0, len(X_shuffled), batch_size):

                # definimos el batch
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                self.a[0] = X_batch     
                self.backpropagation(y_batch, X_batch)
                if self.mejora.get("Rate scheduling lineal", False):
                    self.gradient_descent_rate_scheduling_lineal(epoch)
                elif self.mejora.get("ADAM", False):
                    self.gradient_descent_adam()
                else:
                    self.gradient_descent()

            # la loss con todo X
            y_pred = self.forward_pass(self.X)
            loss = self.cross_entropy_loss(y_pred)
            self.losses.append(loss)

            if X_val is not None and y_val is not None:
                y_val_pred = self.forward_pass(X_val)
                val_loss = self.cross_entropy_loss(y_val_pred, y_val)
                self.losses_val.append(val_loss)



import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, X, y, X_val, y_val, activation_functions: list, nodes_in_layer: list, mejora = None, learning_rate = 0.1, epochs = 1000, graph = True, weights_inicialies = 'He'):
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
        self.graph = graph

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.initialize_weights()

        # mejoras
        if mejora == None:
            self.mejora = {}
        else:
            self.mejora = mejora

        if self.mejora.get("Rate scheduling lineal", False):
            self.lr_min = self.mejora["Rate scheduling lineal"]
        
        if self.mejora.get("ADAM", False):
            # Inicialización para Adam
            self.beta1 = self.mejora["ADAM"][0]
            self.beta2 = self.mejora["ADAM"][1]
            self.epsilon = self.mejora["ADAM"][2]
            self.m_t_weights = [np.zeros_like(w) for w in self.weights]
            self.v_t_weights = [np.zeros_like(w) for w in self.weights]
            self.m_t_biases = [np.zeros_like(b) for b in self.biases]
            self.v_t_biases = [np.zeros_like(b) for b in self.biases]
            self.t = 0
        
        if self.mejora.get("Early stopping", False):
            self.early_stopping_count = self.mejora["Early stopping"]
        
        if self.mejora.get("Dropout", False):
            self.dropout_rates = [self.mejora["Dropout"]]* (self.layers - 2)
            self.dropout_masks = {}

        if self.mejora.get("Batch normalization", False):
            
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
    
    def cross_entropy_loss(self, y_pred, y=None):
        if y is None:
            y = self.y
        else:
            y = np.eye(np.max(y) + 1)[y]
        m = self.X.shape[0]
        loss = -np.sum(y * np.log(y_pred + 1e-8)) / m

        if self.mejora.get("L2", False):
            lambda_ = self.mejora["L2"]
            l2_term = (lambda_ / (2 * m)) * sum(np.sum(w**2) for w in self.weights)
            loss += l2_term

        return loss

    
    def cross_entropy_loss_derivative(self, y_pred, y_true):
        return (y_pred - y_true)

    # def forward_pass(self, a, training = True):
    #     for i in range(self.layers - 1):
    #         self.z[i] = np.dot(a, self.weights[i]) + self.biases[i]
    #         if self.activation_functions[i] == 'ReLU':
    #             a = self.ReLU(self.z[i])
    #         elif self.activation_functions[i] == 'softmax':
    #             a = self.softmax(self.z[i])
            
    #         if training and self.mejora.get("Dropout", False) and i < self.layers - 2:
    #             rate = self.dropout_rates[i]
    #             mask = (np.random.rand(*a.shape) > rate).astype(float)
    #             a *= mask
    #             a /= (1.0 - rate)
    #             self.dropout_masks[i] = mask

    #         self.a[i+1] = a
    #     return a
    def forward_pass(self, a, training=True):
        dropout_masks = {}
        for i in range(self.layers - 1):
            self.z[i] = np.dot(a, self.weights[i]) + self.biases[i]
            
            if self.mejora.get("Batch normalization", False) and i < self.layers - 2:
                self.z[i] = self.batch_norm_forward(self.z[i], i, training)

            if self.activation_functions[i] == 'ReLU':
                a = self.ReLU(self.z[i])
            elif self.activation_functions[i] == 'softmax':
                a = self.softmax(self.z[i])
            
            # Dropout solo si estamos entrenando
            if training and self.mejora.get("Dropout", False) and i < self.layers - 2:
                rate = self.dropout_rates[i]
                mask = (np.random.rand(*a.shape) > rate).astype(float)
                a *= mask
                a /= (1.0 - rate)
                dropout_masks[i] = (mask, rate)

            self.a[i + 1] = a
        
        # si querés usar dropout_masks para el backward, devolvelos como segundo output
        self.last_dropout_masks = dropout_masks  # <- opcional, solo si querés usarlos
        return a

    
    def backward_pass(self, y_pred, y, X) -> None:
        m = X.shape[0]
        L = self.layers - 1
        self.delta[L] = self.cross_entropy_loss_derivative(y_pred, y)

        for i in reversed(range(L)):
            self.gradients_weights[i] = np.dot(self.a[i].T, self.delta[i + 1]) / m
            self.gradients_biases[i] = np.sum(self.delta[i + 1], axis=0, keepdims=True) / m

            if self.mejora.get("L2", False):
                lambda_ = self.mejora["L2"]
                self.gradients_weights[i] += (lambda_ / m) * self.weights[i]

            if i != 0:
                if self.activation_functions[i-1] == 'ReLU':
                    dz = np.dot(self.delta[i + 1], self.weights[i].T)

                    if self.mejora.get("Dropout", False):
                        mask_tuple = self.last_dropout_masks.get(i - 1, None)
                        if mask_tuple is not None:
                            mask, rate = mask_tuple
                            dz *= mask
                            dz /= (1.0 - rate)

                    if self.mejora.get("Batch normalization", False) and i - 1 < len(self.gamma):
                        dz, dgamma, dbeta = self.batch_norm_backward(dz, self.z[i - 1], i - 1)
                        # Guardar gradientes para actualizar gamma y beta si querés
                        self.gradients_gamma[i - 1] = dgamma
                        self.gradients_beta[i - 1] = dbeta

                    self.delta[i] =dz * self.ReLU_derivative(self.z[i-1])              
                else:
                    raise ValueError("Unsupported activation function for backpropagation")
        
    def gradient_descent(self) -> None:
        for i in range(self.layers - 1):
            if self.mejora.get("Batch normalization", False) and i < len(self.gamma):
                self.gamma[i] -= self.learning_rate * self.gradients_gamma[i]
                self.beta[i] -= self.learning_rate * self.gradients_beta[i]
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

            y_pred = self.forward_pass(self.X, training=False)
            loss = self.cross_entropy_loss(y_pred)
            self.losses.append(loss)
            
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward_pass(X_val, training=False)
                val_loss = self.cross_entropy_loss(y_val_pred, y_val)
                self.losses_val.append(val_loss)

                if self.mejora.get("Early stopping", False):
                    # inicializa si no está
                    if not hasattr(self, "best_val_loss"):
                        self.best_val_loss = val_loss
                        self.early_stopping_counter = 0
                    elif val_loss < self.best_val_loss - 1e-4:  # margen mínimo de mejora
                        self.best_val_loss = val_loss
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                        if self.early_stopping_counter >= self.mejora["Early stopping"]:
                            print(f"Early stopping triggered at epoch {epoch}. Best val loss: {self.best_val_loss:.4f}")
                            break


            if self.graph:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss}")
                    print(f"loss val: {val_loss}")
            
        # graph the losses
        if self.graph:
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
        lr_min = self.lr_min # tasa mínima
        decay_ratio = epoch / self.epochs
        current_lr = max(lr_init * (1 - decay_ratio), lr_min)

        for i in range(self.layers - 1):
            self.weights[i] -= current_lr * self.gradients_weights[i]
            self.biases[i] -= current_lr * self.gradients_biases[i]

    def gradient_descent_adam(self, minibatch = False):
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

    def fit_mini_batch(self, X_val, y_val, graph = True) -> None:
        batch_size = self.mejora.get("Mini batch stochastic gradient descent")
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
                    self.gradient_descent_adam(minibatch = True)
                else:
                    self.gradient_descent()

            # la loss con todo X
            y_pred = self.forward_pass(self.X, training=False)
            loss = self.cross_entropy_loss(y_pred)
            self.losses.append(loss)

            if X_val is not None and y_val is not None:
                y_val_pred = self.forward_pass(X_val, training = False)
                val_loss = self.cross_entropy_loss(y_val_pred, y_val)
                self.losses_val.append(val_loss)

            if self.graph:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss}")
                    print(f"loss val: {val_loss}")
        if self.graph:
            self.graph_losses()

    def batch_norm_forward(self, z, layer_index, training=True):
        
    
    def batch_norm_backward(self, dz, z, layer_index):
        

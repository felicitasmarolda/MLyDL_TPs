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
        
        if self.mejora.get("Rate scheduling exponencial", False):
            self.decay_rate = self.mejora["Rate scheduling exponencial"]
        
        if self.mejora.get("ADAM", False):
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
        if y.ndim == 1 or y.shape[1] == 1:
            y = np.eye(y_pred.shape[1])[y.reshape(-1).astype(int)]

        m = self.X.shape[0]
        loss = -np.sum(y * np.log(y_pred + 1e-8)) / m

        if self.mejora.get("L2", False):
            lambda_ = self.mejora["L2"]
            l2_term = (lambda_ / (2 * m)) * sum(np.sum(w**2) for w in self.weights)
            loss += l2_term

        return loss

    
    def cross_entropy_loss_derivative(self, y_pred, y_true):
        return (y_pred - y_true)


    def forward_pass(self, a, training=True):
        dropout_masks = {}
        for i in range(self.layers - 1):
            self.z[i] = np.dot(a, self.weights[i]) + self.biases[i]
            
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
        
        self.last_dropout_masks = dropout_masks 
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
                if self.mejora.get("Rate scheduling exponencial", False):
                    current_lr = self.learning_rate * (self.decay_rate ** epoch)
                    self.gradient_descent_adam(current_lr)
                else:
                    self.gradient_descent_adam()
            elif self.mejora.get("Rate scheduling exponencial", False):
                self.gradient_descent_rate_scheduling_exponencial(epoch)
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
                    # si esta subiendo hace patience epochs corto
                    if epoch > 10 and (val_loss > self.losses_val[-2] or val_loss > self.losses_val[-3]):
                        # si la validacion no mejora en patience epochs, se corta
                        self.early_stopping_count -= 1
                        # print(f"Early stopping count: {self.early_stopping_count} in epoch {epoch}")
                        if self.early_stopping_count == 0:
                            print("Early stopping triggered")
                            break
                    else:
                        self.early_stopping_count = self.mejora["Early stopping"]


            if self.graph:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss}")
                    print(f"loss val: {val_loss}")
            
        if self.graph:
            self.graph_losses()

    def graph_losses(self):
        fs = 13
        plt.plot(self.losses, label='Train Loss', color='limegreen')
        if self.losses_val:
            plt.plot(self.losses_val, label='Validation Loss', color='deeppink')
        plt.xlabel('Epochs', fontsize=fs)
        plt.ylabel('Cross entropy Loss', fontsize=fs)
        plt.title('Loss vs Epochs', fontsize=fs)
        plt.legend(fontsize=fs)
        plt.show()

    # mejoras
    def gradient_descent_rate_scheduling_lineal(self, epoch):
        lr_init = self.learning_rate
        lr_min = self.lr_min
        decay_ratio = epoch / self.epochs
        current_lr = max(lr_init * (1 - decay_ratio), lr_min)

        for i in range(self.layers - 1):
            self.weights[i] -= current_lr * self.gradients_weights[i]
            self.biases[i] -= current_lr * self.gradients_biases[i]

    def gradient_descent_rate_scheduling_exponencial(self, epoch):
        current_lr = self.learning_rate * (self.decay_rate ** epoch)
        for i in range(self.layers - 1):
            self.weights[i] -= current_lr * self.gradients_weights[i]
            self.biases[i] -= current_lr * self.gradients_biases[i]


    def gradient_descent_adam(self, current_lr=None):
        self.t += 1 
        
        lr = current_lr if current_lr is not None else self.learning_rate
        
        for i in range(self.layers - 1):
            self.m_t_weights[i] = self.beta1 * self.m_t_weights[i] + (1 - self.beta1) * self.gradients_weights[i]
            self.v_t_weights[i] = self.beta2 * self.v_t_weights[i] + (1 - self.beta2) * (self.gradients_weights[i]**2)
            
            m_hat_w = self.m_t_weights[i] / (1 - self.beta1**self.t)
            v_hat_w = self.v_t_weights[i] / (1 - self.beta2**self.t)
            
            self.weights[i] -= lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            
            self.m_t_biases[i] = self.beta1 * self.m_t_biases[i] + (1 - self.beta1) * self.gradients_biases[i]
            self.v_t_biases[i] = self.beta2 * self.v_t_biases[i] + (1 - self.beta2) * (self.gradients_biases[i]**2)
            
            m_hat_b = self.m_t_biases[i] / (1 - self.beta1**self.t)
            v_hat_b = self.v_t_biases[i] / (1 - self.beta2**self.t)
            
            self.biases[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
    
    def fit_mini_batch(self, X_val, y_val, graph = True) -> None:
        batch_size = self.mejora.get("Mini batch stochastic gradient descent")
        for epoch in range(self.epochs):
            
            indices = np.arange(self.X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]

            for i in range(0, len(X_shuffled), batch_size):

                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                self.a[0] = X_batch     
                self.backpropagation(y_batch, X_batch)
                if self.mejora.get("Rate scheduling lineal", False):
                    self.gradient_descent_rate_scheduling_lineal(epoch)
                elif self.mejora.get("Rate scheduling exponencial", False):
                    self.gradient_descent_rate_scheduling_exponencial(epoch)
                elif self.mejora.get("ADAM", False):
                    if self.mejora.get("Rate scheduling exponencial", False):
                        current_lr = self.learning_rate * (self.decay_rate ** epoch)
                        self.gradient_descent_adam(current_lr)
                    else:
                        self.gradient_descent_adam()
                else:
                    self.gradient_descent()

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

        
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.init as init

class NNTorch(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, mejora=None):
        super(NNTorch, self).__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.mejora = mejora if mejora else {}

        self.layers = nn.ModuleList()
        
        # Input
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        self.layers.append(nn.ReLU())
        
        # Dropout
        if self.mejora.get("Dropout", False):
            self.layers.append(nn.Dropout(self.mejora["Dropout"]))
        
        # Ocultas
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.layers.append(nn.ReLU())

            if self.mejora.get("Dropout", False):
                self.layers.append(nn.Dropout(self.mejora["Dropout"]))

        # Output
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))
        
        # L2
        self.l2_lambda = self.mejora.get("L2", 0.0)
        
        self.train_losses = []
        self.val_losses = []

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='relu' if layer != self.layers[-1] else 'linear')
                init.zeros_(layer.bias)

    def forward(self, x):
        out = x
        for layer in self.layers:
            # print(layer)
            out = layer(out)
        return out

    def fit(self, X, y, X_val=None, y_val=None, learning_rate=0.01, epochs=1000, batch_size=None, graph=True):
        if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.LongTensor(y)

        dataset = TensorDataset(X, y)
        use_minibatch = self.mejora.get("MiniBatch", False)

        if use_minibatch:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            dataloader = [(X, y)]  # Entreno con todo el dataset

        
        if isinstance(X_val, np.ndarray):
            X_val = torch.FloatTensor(X_val)
        if isinstance(y_val, np.ndarray):
            y_val = torch.LongTensor(y_val)
        
        # funcion de loss
        criterion = nn.CrossEntropyLoss()
        
        # optimizados para learning rate
        if self.mejora.get("ADAM", False):
            beta1, beta2, epsilon = self.mejora["ADAM"]
            optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon, weight_decay=self.l2_lambda)
        else:
            optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=self.l2_lambda)
        
        scheduler = None
        if self.mejora.get("Rate scheduling exponencial", False):
            decay_rate = self.mejora["Rate scheduling exponencial"]
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        elif self.mejora.get("Rate scheduling lineal", False):
            lr_min = self.mejora["Rate scheduling lineal"]
            lambda_fn = lambda epoch: max(1.0 - epoch/epochs, lr_min/learning_rate)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)
            
        # Early stopping
        best_val_loss = float('inf')
        early_stopping_counter = 0
        early_stopping_patience = self.mejora.get("Early stopping", float('inf'))
        
        # entrenamiento
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            
            # Mini-batch training
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
            
            epoch_loss = total_loss / len(dataset)
            self.train_losses.append(epoch_loss)
            
            val_loss = None
            val_loss = self._evaluate(X_val, y_val, criterion)
            self.val_losses.append(val_loss)
            
            if self.mejora.get("Early stopping", False):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        if graph:
                            print(f"Early stopping triggered at epoch {epoch}")
                        break
            
            if scheduler:
                scheduler.step()
            
            if graph and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}", end="")
                if val_loss:
                    print(f", Val Loss: {val_loss:.4f}")
                else:
                    print()
        
        if graph:
            self._graph_losses()
    
    def _evaluate(self, X, y, criterion):
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            loss = criterion(outputs, y)
        return loss.item()
    
    def _graph_losses(self):
        fs = 13
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='limegreen')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', color='deeppink')
        # plt.yscale('log')
        plt.xlabel('Epochs', fontsize=fs)
        plt.ylabel('Cross Entropy Loss', fontsize=fs)
        plt.title('Loss vs Epochs', fontsize=fs)
        plt.legend(fontsize=fs)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def predict_class(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        self.eval()
        
        with torch.no_grad():
            outputs = self(X)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.numpy()
    
    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        self.eval()
        
        with torch.no_grad():
            outputs = self(X)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        return probabilities.numpy()
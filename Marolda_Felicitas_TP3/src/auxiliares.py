import numpy as np
import models as md
import metricas as mt

def cross_validation_lr(X, y, learning_rates, params:tuple, k = 5):
    n = X.shape[0]
    fold_size = n // k
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    results = []
    for lr in learning_rates:
        print(f"Learning Rate: {lr}")
        fold_results = []
        for i in range(k):
            print(f"Fold {i+1}/{k}")
            val_indices = folds[i]
            train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            
            model = md.NeuralNetwork(X_train, y_train, X_val, y_val, params[0], params[1], mejora = params[2], learning_rate = lr, epochs = 1000, graph = False)

            y_pred = model.forward_pass(X_val, False)
            y_pred_labels = np.argmax(y_pred, axis=1)

            acc = mt.accuracy(y_pred_labels, y_val)
            ce = mt.cross_entropy(y_val, y_pred)

            fold_results.append((acc, ce))
        avg_acc = np.mean([result[0] for result in fold_results])
        avg_ce = np.mean([result[1] for result in fold_results])
        results.append((avg_acc, avg_ce))
    return results
            

def cross_validation_sgd(X, y, batch_sizes, params, k = 5):
    n = X.shape[0]
    fold_size = n // k
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    results = []
    for bs in batch_sizes:
        print(f"Batch size: {bs}")
        fold_results = []
        for i in range(k):
            print(f"Fold {i+1}/{k}")
            val_indices = folds[i]
            train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            
            mejoras = {}
            mejoras["Mini batch stochastic gradient descent"] = bs
            mejoras["Early stopping"] = 5
            model = md.NeuralNetwork(X_train, y_train, X_val, y_val, params[0], params[1], mejora = mejoras, learning_rate = params[2], epochs = 1000, graph = False)

            y_pred = model.forward_pass(X_val, False)
            y_pred_labels = np.argmax(y_pred, axis=1)

            acc = mt.accuracy(y_pred_labels, y_val)
            ce = mt.cross_entropy(y_val, y_pred)

            fold_results.append((acc, ce))
        avg_acc = np.mean([result[0] for result in fold_results])
        avg_ce = np.mean([result[1] for result in fold_results])
        print("Avg results: \n"," "*20, "avg acc = ", avg_acc, " "*5, "avg ce = ", avg_ce)
        results.append((avg_acc, avg_ce))
    return results

def cross_validation_mejora(X, y, nombre, hiperparametros, params, k = 5):
    n = X.shape[0]
    fold_size = n // k
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    results = []
    for hp in hiperparametros:
        print(f"hp: {hp}")
        fold_results = []
        for i in range(k):
            # print(f"Fold {i+1}/{k}")
            # Split data into training and validation sets
            val_indices = folds[i]
            train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            
            mejoras = {}
            mejoras[nombre] = hp
            mejoras["Early stopping"] = 5

            model = md.NeuralNetwork(X_train, y_train, X_val, y_val, params[0], params[1], mejora = mejoras, learning_rate = params[2], epochs = 1000, graph = False)

            y_pred = model.forward_pass(X_val, False)
            y_pred_labels = np.argmax(y_pred, axis=1)

            acc = mt.accuracy(y_pred_labels, y_val)
            ce = mt.cross_entropy(y_val, y_pred)

            fold_results.append((acc, ce))
        avg_acc = np.mean([result[0] for result in fold_results])
        avg_ce = np.mean([result[1] for result in fold_results])
        print("Avg results: \n"," "*20, "avg acc = ", avg_acc, " "*5, "avg ce = ", avg_ce)
        results.append((avg_acc, avg_ce))
    return results


def cross_validation(X, y, params, k = 5):
    n = X.shape[0]
    fold_size = n // k
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    results = []

    for i in range(k):
        val_indices = folds[i]
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        model = md.NeuralNetwork(X_train, y_train, X_val, y_val, params[0], params[1], mejora = params[3], learning_rate = params[2], epochs = 1000, graph = False)

        y_pred = model.forward_pass(X_val, False)
        y_pred_labels = np.argmax(y_pred, axis=1)

        acc = mt.accuracy(y_pred_labels, y_val)
        print("acc: ", acc)
        ce = mt.cross_entropy(y_val, y_pred)
        print("ce: ", ce)

        results.append((acc, ce))

    avg_acc = np.mean([r[0] for r in results])
    avg_ce = np.mean([r[1] for r in results])

    return results, avg_acc, avg_ce


def graph(x,y1, y2, title, xlabel, names, scale = None, unite = False):
    import matplotlib.pyplot as plt

    fs = 13

    plt.scatter(x, y1, label=names[0], color='cornflowerblue')
    plt.scatter(x, y2, label=names[1], color='indigo')

    if unite:
        plt.plot(x, y1, label=names[0], color='cornflowerblue')
        plt.plot(x, y2, label=names[1], color='indigo')

    plt.xlabel(xlabel, fontsize=fs)
    plt.ylabel('Accuracy / Cross Entropy', fontsize=fs)
    plt.legend(fontsize=fs)

    if scale == 'log':
        plt.xscale('log')

    plt.show()
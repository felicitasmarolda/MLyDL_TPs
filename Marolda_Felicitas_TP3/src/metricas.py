import numpy as np
import matplotlib.pyplot as plt

def accuracy(y_pred, y_true):
    y_true = np.ravel(y_true).copy()
    y_pred = np.ravel(y_pred).copy()
    return np.sum(y_true == y_pred) / len(y_true)

def confusion_matrix_multiclass(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    matriz = np.zeros((len(classes), len(classes)), dtype=int)
    for i, true in enumerate(classes):
        for j, pred in enumerate(classes):
            matriz[i, j] = np.sum((y_true == true) & (y_pred == pred))

    return matriz

def cross_entropy(y_true, y_pred):
    """Compute cross entropy loss for multiclass classification"""
    # Asegurarse de que y_true sea un vector de etiquetas, no one-hot
    y_true = np.asarray(y_true).astype(int)
    # Clipping para evitar log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.log(y_pred[np.arange(len(y_true)), y_true]))

def get_metrics(y_pred, y_true):
    """ print all metrics and show confusion matrix """
    
    y_pred_labels = np.argmax(y_pred, axis=1)

    acc = accuracy(y_pred_labels, y_true)
    ce = cross_entropy(y_true, y_pred)  # aquí sí necesitas las probabilidades
    cm = confusion_matrix_multiclass(y_true, y_pred_labels)

    print(f'Accuracy: {acc:.4f}')
    print(f'Cross Entropy: {ce:.4f}')

    plt.imshow(cm, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def print_acc_ce(y_pred, y_true):
    """ print accuracy and cross entropy """
    
    y_pred_labels = np.argmax(y_pred, axis=1)

    acc = accuracy(y_pred_labels, y_true)
    ce = cross_entropy(y_true, y_pred)  # aquí sí necesitas las probabilidades

    print(f'Accuracy: {acc:.4f}')
    print(f'Cross Entropy: {ce:.4f}')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix_multiclass(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    matriz = np.zeros((len(classes), len(classes)), dtype=int)
    for i, true in enumerate(classes):
        for j, pred in enumerate(classes):
            matriz[i, j] = np.sum((y_true == true) & (y_pred == pred))

    return matriz

def draw_confusion_matrix_multiclass(y_true, y_pred):
    y_true_copy = np.ravel(y_true).copy()
    y_pred_copy = np.ravel(y_pred).copy()
    matriz = confusion_matrix_multiclass(y_true, y_pred)
    classes = np.unique(np.concatenate((y_true_copy, y_pred_copy)))
    df_cm = pd.DataFrame(matriz, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def get_trues_and_falses(y_true, y_pred, label):
    y_true = np.ravel(y_true).copy()
    y_pred = np.ravel(y_pred).copy()

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(y_true)):
        if y_true[i] == label and y_pred[i] == label:
            TP += 1
        elif y_true[i] == label and y_pred[i] != label:
            FN += 1
        elif y_true[i] != label and y_pred[i] == label:
            FP += 1
        else:
            TN += 1

    return TP, TN, FP, FN

def accuracy(y_true, y_pred):
    y_true = np.ravel(y_true).copy()
    y_pred = np.ravel(y_pred).copy()
    return np.sum(y_true == y_pred) / len(y_true)

def precision_multiclass(y_true, y_pred):
    y_true = np.ravel(y_true).copy()
    y_pred = np.ravel(y_pred).copy()
    precisions = []
    classes = np.unique(np.concatenate((y_true, y_pred)))
    for label in classes:
        TP, TN, FP, FN = get_trues_and_falses(y_true, y_pred, label)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        precisions.append(precision)
    return precisions

def recall_multiclass(y_true, y_pred):
    y_true = np.ravel(y_true).copy()
    y_pred = np.ravel(y_pred).copy()
    recalls = []
    classes = np.unique(np.concatenate((y_true, y_pred)))
    for label in classes:
        TP, TN, FP, FN = get_trues_and_falses(y_true, y_pred, label)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        recalls.append(recall)
    return recalls

def f_score_multiclass(y_true, y_pred, beta=1):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    p = np.array(precision_multiclass(y_true, y_pred))
    r = np.array(recall_multiclass(y_true, y_pred))
    numerator = (1 + beta**2) * (p * r)
    denominator = (beta**2 * p + r)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return f1


def curve_precision_recall(y_true, y_scores, label):
    thresholds = np.arange(0, 1.0, 0.1)
    precisions = []
    recalls = []

    binary_true = (y_true == label).astype(int)

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        TP, TN, FP, FN = get_trues_and_falses(binary_true, y_pred, label=1)
        if (TP + FP) > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0

        if (TP + FN) > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

def AUC_PR(y_true, y_proba):
    y_true = np.ravel(y_true)
    classes = np.unique(y_true)
    aucs = []

    class_to_index = {label: idx for idx, label in enumerate(classes)}
    for label in classes:
        idx = class_to_index[label]
        scores = y_proba[:, idx]
        precisions, recalls = curve_precision_recall(y_true, scores, label)
        auc = np.trapz(precisions, recalls)
        aucs.append(auc)

    return np.mean(aucs)  # macro-promedio


def curve_ROC_multiclass(y_true, y_proba):
    y_true = np.ravel(y_true)
    classes = np.unique(y_true)
    roc_data = {}

    class_to_index = {label: idx for idx, label in enumerate(classes)}

    for label in classes:
        binary_true = (y_true == label).astype(int)
        idx = class_to_index[label]
        scores = y_proba[:, idx]  # ahora sí el índice es válido

        TPRs = []
        FPRs = []
        thresholds = np.arange(0, 1.01, 0.1)
        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(int)
            TP, TN, FP, FN = get_trues_and_falses(binary_true, y_pred, label=1)
            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            TPRs.append(TPR)
            FPRs.append(FPR)

        roc_data[label] = (FPRs, TPRs)

    return roc_data



def AUC_ROC_multiclass(y_true, y_proba):
    roc_data = curve_ROC_multiclass(y_true, y_proba)
    aucs = []

    for label, (FPRs, TPRs) in roc_data.items():
        auc = np.trapz(TPRs, FPRs)
        aucs.append(auc)

    return np.mean(aucs)  # macro-promedio


def graph_val_fscore(val_list, fscores):
    """Graficamos el valor de L2 contra su fscore correspondiente"""
    plt.figure()
    plt.plot(val_list, fscores, marker='o')
    plt.xlabel('L2')
    plt.ylabel('Fscore')
    plt.title('L2 vs Fscore')
    plt.xscale('log')
    plt.grid(True)
    plt.show()
    plt.savefig('L2_vs_Fscore.png')
    plt.close()
    
def get_metrics_multiclass(y_true, y_pred, y_proba):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    y_proba = np.array(y_proba)  # debe ser (n_samples, n_classes)

    # metricas
    acc = accuracy(y_true, y_pred)
    prec = precision_multiclass(y_true, y_pred)
    rec = recall_multiclass(y_true, y_pred)
    f1 = f_score_multiclass(y_true, y_pred)
    auc_roc = AUC_ROC_multiclass(y_true, y_proba)
    auc_pr = AUC_PR(y_true, y_proba)

    # mostrar tabla con pandas
    metrics_df = pd.DataFrame({
        "Accuracy": [acc],
        "Precision": [prec],
        "Recall": [rec],
        "F1-score": [f1],
        "AUC-ROC": [auc_roc],
        "AUC-PR": [auc_pr]
    })
    metrics_df.index = ["Metrics"]
    metrics_df.to_csv("metrics.csv", index=True)

    # calcular curvas ROC
    roc_data = curve_ROC_multiclass(y_true, y_proba)

    # confusin matriz
    matriz = confusion_matrix_multiclass(y_true, y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    df_cm = pd.DataFrame(matriz, index=classes, columns=classes)
    plt.figure(figsize=(14, 7))

    # Confusion Matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # ROC curve por clase
    plt.subplot(1, 3, 2)
    for label, (FPRs, TPRs) in roc_data.items():
        plt.plot(FPRs, TPRs, marker='o', label=f"Clase {label}")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend(loc="lower right")

    # Precision-Recall curve por clase
    # Precision-Recall curve por clase
    plt.subplot(1, 3, 3)
    classes = np.unique(y_true)
    for i, label in enumerate(classes):
        precisions, recalls = curve_precision_recall(y_true, y_proba[:, i], label=label)
        plt.plot(recalls, precisions, marker='o', label=f"Clase {label}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()




def get_numeric_metrics_multiclass(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    acc = accuracy(y_true, y_pred)
    prec = precision_multiclass(y_true, y_pred)
    rec = recall_multiclass(y_true, y_pred)
    f1 = f_score_multiclass(y_true, y_pred)

    return [acc, prec, rec, f1]

def graph_all_metrics_multiclass(numeric_sr, numeric_us, numeric_od, numeric_smote, numeric_cr):
    """
    Graficamos Accuracy, Precision, Recall y F1 para cada estrategia multiclase
    """
    metrics_df = pd.DataFrame({
        "SR": numeric_sr,
        "US": numeric_us,
        "OD": numeric_od,
        "SMOTE": numeric_smote,
        "CR": numeric_cr
    }, index=["Accuracy", "Precision", "Recall", "F1-score"]).T

    metrics_df.plot(kind='bar', figsize=(10, 6))
    plt.title("Comparación de métricas multiclase por técnica de rebalanceo")
    plt.ylabel("Valor")
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    metrics_df.to_csv("metrics_multiclass.csv", index=True)

    return metrics_df

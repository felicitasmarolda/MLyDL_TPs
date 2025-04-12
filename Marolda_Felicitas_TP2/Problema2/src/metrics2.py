import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix_multiclass(y_true, y_pred, label=None):
    """
    Matriz de confusión ara multiclase
    """
    TP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
    FP = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
    FN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
    TN = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp != label)
    return TP, TN, FP, FN


def draw_confusion_matrix_multiclass(TP, TN, FP, FN):
    matrix = np.array([[TN, FP],
                       [FN, TP]])

    # Plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix_multiclass(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def precision_multiclass(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    
    labels = np.unique(np.concatenate((y_true, y_pred)))
    precisions = []
    for label in labels:
        TP, TN, FP, FN  = confusion_matrix_multiclass(y_true, y_pred, label)
        if TP + FP == 0:
            precisions.append(0)
        else:
            precisions.append(TP / (TP + FP))
    return np.mean(precisions)


def recall_multiclass(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    recalls = []
    for label in labels:
        TP, TN, FP, FN = confusion_matrix_multiclass(y_true, y_pred, label)
        if TP + FN == 0:
            recalls.append(0)
        else:
            recalls.append(TP / (TP + FN))
    return np.mean(recalls)


def f_score_multiclass(y_true, y_pred, beta=1):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    p = precision_multiclass(y_true, y_pred)
    r = recall_multiclass(y_true, y_pred)
    if (beta**2 * p + r) == 0:
        return 0
    return (1 + beta**2) * (p * r) / (beta**2 * p + r)

def curve_precision_recall(y_true, y_scores):
    thresholds = np.arange(0, 1.01, 0.1)  # mejor usar 1.01 para incluir 1.0 por seguridad
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        TP, TN, FP, FN = confusion_matrix_multiclass(y_true, y_pred)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


def draw_precision_recall_curve(y_true, y_scores):
    precisions, recalls = curve_precision_recall(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.show()

def curve_ROC(y_true, y_scores):
    # print(np.unique(y_scores))
    thresholds = np.arange(0, 1.01, 0.1)  # mejor usar 1.01 para incluir 1.0 por seguridad
    TPRs = []
    FPRs = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        TP, TN, FP, FN = confusion_matrix_multiclass(y_true, y_pred)
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        TPRs.append(TPR)
        FPRs.append(FPR)

    return FPRs, TPRs


def draw_ROC_curve(y_true, y_scores):
    FPRs, TPRs = curve_ROC(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(FPRs, TPRs, marker='o', label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')  # línea de referencia
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend()
    plt.show()


def AUC_ROC(y_true, y_scores):
    FPRs, TPRs = curve_ROC(y_true, y_scores)
    auc = np.trapz(TPRs, FPRs)
    return auc

def draw_AUC_ROC(y_true, y_scores):
    FPRs, TPRs = curve_ROC(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(FPRs, TPRs, marker='o')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.fill_between(FPRs, TPRs, alpha=0.2)
    plt.show()
    

def AUC_PR(y_true, y_scores):
    precisions, recalls = curve_precision_recall(y_true, y_scores)
    auc = np.trapz(precisions, recalls)
    return auc

def draw_AUC_PR(y_true, y_scores):
    precisions, recalls = curve_precision_recall(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUC-PR Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.fill_between(recalls, precisions, alpha=0.2)
    plt.show()

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
    
def get_metrics_multiclass(y_true, y_proba, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    y_proba = np.ravel(y_proba)

    # metricas
    acc = accuracy(y_true, y_pred)
    prec = precision_multiclass(y_true, y_pred)
    rec = recall_multiclass(y_true, y_pred)
    f1 = f_score_multiclass(y_true, y_pred)
    auc_roc = AUC_ROC(y_true, y_proba)
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

    # calcular curvas
    FPRs, TPRs = curve_ROC(y_true, y_proba)
    precisions, recalls = curve_precision_recall(y_true, y_proba)

    # confusin matriz
    TP, TN, FP, FN = confusion_matrix_multiclass(y_true, y_pred, label=1)
    

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

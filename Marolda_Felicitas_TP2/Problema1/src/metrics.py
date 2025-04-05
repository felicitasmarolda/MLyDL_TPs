import numpy as np
import pandas as pd

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP, TN, FP, FN

def accuracy(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def precision(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f_score(y_true, y_pred, beta=1):
    print("netro")
    p = precision(y_true, y_pred)
    print(f"Precision: {p}")
    r = recall(y_true, y_pred)
    print(f"Recall: {r}")
    return (1 + beta**2) * (p * r) / (beta**2 * p + r) if (beta**2 * p + r) > 0 else 0

def curve_precision_recall(y_true, y_scores):
    thresholds = np.arange(0, 1.1, 0.1)
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        precisions.append(precision(y_true, y_pred))
        recalls.append(recall(y_true, y_pred))

    return precisions, recalls

def curve_ROC(y_true, y_scores):
    thresholds = np.arange(0, 1.1, 0.1)
    TPRs = []  # True Positive Rates
    FPRs = []  # False Positive Rates

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
        TPRs.append(recall(y_true, y_pred))  # Sensitivity
        FPRs.append(FP / (FP + TN) if (FP + TN) > 0 else 0)  # Fall-out

    return FPRs, TPRs

def AUC_ROC(y_true, y_scores):
    FPRs, TPRs = curve_ROC(y_true, y_scores)
    auc = np.trapz(TPRs, FPRs)
    return auc

def AUC_PR(y_true, y_scores):
    precisions, recalls = curve_precision_recall(y_true, y_scores)
    auc = np.trapz(precisions, recalls)
    return auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix and display it.
    """
    TP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    TN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    FP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    FN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    return TP, TN, FP, FN

def draw_confusion_matrix(TP, TN, FP, FN):
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
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def precision(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f_score(y_true, y_pred, beta=1):
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

def draw_precision_recall_curve(y_true, y_scores):
    precisions, recalls = curve_precision_recall(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.show()

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

def draw_ROC_curve(y_true, y_scores):
    FPRs, TPRs = curve_ROC(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(FPRs, TPRs, marker='o')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
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
    plt.title('ROC Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.fill_between(FPRs, TPRs, alpha=0.2)
    plt.show()
    

def AUC_PR(y_true, y_scores):

    precisions, recalls = curve_precision_recall(y_true, y_scores)
    auc = np.trapz(precisions, recalls)
    return auc


def graph_L2_fscore(L2_list, fscores):
    """Graficamos el valor de L2 contra su fscore correspondiente"""
    plt.figure()
    plt.plot(L2_list, fscores, marker='o')
    plt.xlabel('L2')
    plt.ylabel('Fscore')
    plt.title('L2 vs Fscore')
    plt.xscale('log')
    plt.grid(True)
    plt.show()
    plt.savefig('L2_vs_Fscore.png')
    plt.close()
    
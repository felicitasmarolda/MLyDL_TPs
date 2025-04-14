import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix(y_true, y_pred):
    TP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    TN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    FP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    FN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    return TP, TN, FP, FN

def draw_confusion_matrix(TP, TN, FP, FN):
    matrix = np.array([[TN, FP],
                       [FN, TP]])

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
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall(y_true, y_pred):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f_score(y_true, y_pred, beta=1):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1 + beta**2) * (p * r) / (beta**2 * p + r) if (beta**2 * p + r) > 0 else 0

def curve_precision_recall(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
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
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

def curve_ROC(y_true, y_scores):
    thresholds = np.arange(0, 1.01, 0.1)
    TPRs = []
    FPRs = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        TPRs.append(TPR)
        FPRs.append(FPR)
    
    FPRs, TPRs = zip(*sorted(zip(FPRs, TPRs), key=lambda x: x[0]))

    return FPRs, TPRs

def draw_ROC_curve(y_true, y_scores):
    FPRs, TPRs = curve_ROC(y_true, y_scores)
    print("FPRs:", FPRs)
    print("TPRs:", TPRs)
    plt.figure(figsize=(8, 6))
    plt.plot(FPRs, TPRs, marker='o', label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier') 
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
    
def AUC_PR(y_true, y_scores):
    precisions, recalls = curve_precision_recall(y_true, y_scores)
    auc = np.trapz(precisions, recalls)
    return auc

def graph_val_fscore(val_list, fscores):
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
    
def get_metrics(y_true, y_pred, y_proba, threshold=0.5):

    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    y_proba = np.ravel(y_proba)
    # print("y_true:", len(y_true))
    # print("y_pred:", len(y_pred))
    # print("y_proba:", len(y_proba))

    # Métricas
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f_score(y_true, y_pred)
    auc_roc = AUC_ROC(y_true, y_proba)
    auc_pr = AUC_PR(y_true, y_proba)

    # Mostrar tabla con pandas
    metrics_df = pd.DataFrame({
        "Accuracy": [acc],
        "Precision": [prec],
        "Recall": [rec],
        "F1-score": [f1],
        "AUC-ROC": [auc_roc],
        "AUC-PR": [auc_pr]
    })
    print("===== MÉTRICAS =====")
    print(metrics_df.to_string(index=False))

    # Calcular curvas
    FPRs, TPRs = curve_ROC(y_true, y_proba)
    precisions, recalls = curve_precision_recall(y_true, y_proba)
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    # print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
    matrix = np.array([[TN, FP], [FN, TP]])

    # === FIGURA ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion Matrix
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Pred Neg', 'Pred Pos'],
                yticklabels=['Actual Neg', 'Actual Pos'],
                ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # ROC Curve
    # print("FPRs:", FPRs)
    # print("TPRs:", TPRs)
    axes[1].plot(FPRs, TPRs, label='ROC Curve')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].grid(True)
    axes[1].legend()

    # PR Curve
    axes[2].plot(recalls, precisions)
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve')
    axes[2].grid(True)
    plt.rcParams.update({
        "font.size": 15,
    })
    plt.tight_layout()
    plt.show()

    return metrics_df

def get_numeric_metrics(y_true, y_scores, y_proba, threshold=0.5):
    y_true = np.ravel(y_true)
    y_scores = np.ravel(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    # Métricas
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f_score(y_true, y_pred)
    auc_roc = AUC_ROC(y_true, y_proba)
    auc_pr = AUC_PR(y_true, y_proba)

    return [acc, prec, rec, f1, auc_roc, auc_pr]

def get_metrics_for_graphing(y_true, y_scores, y_proba):
    y_true = np.ravel(y_true)
    y_scores = np.ravel(y_scores)

    # Calcular curvas
    FPRs, TPRs = curve_ROC(y_true, y_proba)
    precisions, recalls = curve_precision_recall(y_true, y_proba)

    return [FPRs, TPRs, precisions, recalls]

def graph_all_metrics_rebalanced(numeric_sr, numeric_us, numeric_od, numeric_smote, numeric_cr, graphing_sr, graphing_us, graphing_od, graphing_smote, graphing_cr):
    # Convertir a numpy arrays
    numeric_sr = np.array(numeric_sr)
    numeric_us = np.array(numeric_us)
    numeric_od = np.array(numeric_od)
    numeric_smote = np.array(numeric_smote)
    numeric_cr = np.array(numeric_cr)

    TPRs = []
    FPRs = []
    precisions = []
    recalls = []

    TPRs.append(graphing_sr[1])
    TPRs.append(graphing_us[1])
    TPRs.append(graphing_od[1])
    TPRs.append(graphing_smote[1])
    TPRs.append(graphing_cr[1])
    FPRs.append(graphing_sr[0])
    FPRs.append(graphing_us[0])
    FPRs.append(graphing_od[0])
    FPRs.append(graphing_smote[0])
    FPRs.append(graphing_cr[0])

    precisions.append(graphing_sr[2])
    precisions.append(graphing_us[2])
    precisions.append(graphing_od[2])
    precisions.append(graphing_smote[2])
    precisions.append(graphing_cr[2])
    recalls.append(graphing_sr[3])
    recalls.append(graphing_us[3])
    recalls.append(graphing_od[3])
    recalls.append(graphing_smote[3])
    recalls.append(graphing_cr[3])


    # Graficar
    # Plot Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot Precision-Recall Curves
    for i, (precision, recall, label) in enumerate(zip(precisions, recalls, ["SR", "US", "OD", "SMOTE", "CR"])):
        axes[0].plot(recall, precision, label=f"{label} PR Curve")
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curves')
    axes[0].legend()
    axes[0].grid(True)

    # Plot ROC Curves
    for i, (fpr, tpr, label) in enumerate(zip(FPRs, TPRs, ["SR", "US", "OD", "SMOTE", "CR"])):
        axes[1].plot(fpr, tpr, label=f"{label} ROC Curve")
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')  # Reference line
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curves')
    axes[1].legend()
    axes[1].grid(True)

    plt.rcParams.update({
    "font.size": 25,           # tamaño general de fuente
})
    plt.tight_layout()
    plt.show()

    # Hacer una tabla con las métricas
    metrics_df = pd.DataFrame({
        "SR": numeric_sr,
        "US": numeric_us,
        "OD": numeric_od,
        "SMOTE": numeric_smote,
        "CR": numeric_cr
    }, index=["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC", "AUC-PR"]).T


    # print("===== MÉTRICAS =====")
    # print(metrics_df.to_string(index=True))
    metrics_df.to_csv('metrics.csv', index=True)

    return metrics_df
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


def curve_precision_recall(y_true, y_scores, thresholds_pr, label):
    precisions = []
    recalls = []

    binary_true = (y_true == label).astype(int)

    for threshold in thresholds_pr:
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
    # print(f"precisions: {precisions}")
    # print(f"recalls: {recalls}")

    return precisions, recalls

def AUC_PR_multiclass(y_true, y_proba, thresholds_pr):
    y_true = np.ravel(y_true)
    classes = np.unique(y_true)
    aucs = []

    classes = np.unique(y_true)
    for i, label in enumerate(classes):
        precisions, recalls = curve_precision_recall(y_true, y_proba[:, i], thresholds_pr, label)
        points = sorted(zip(recalls, precisions), key=lambda x: x[0])
        recalls = [r for r, p in points]
        precisions = [p for r, p in points]
        
        if recalls[0] != 0:
            recalls = [0] + recalls
            precisions = [precisions[0]] + precisions
        recalls, precisions = zip(*sorted(zip(recalls, precisions)))

        auc = np.trapz(precisions, recalls)

        aucs.append(auc)

    return np.mean(aucs)

def curve_ROC_multiclass(y_true, y_proba, thresholds):
    y_true = np.ravel(y_true)
    classes = np.unique(y_true)
    roc_data = {}

    class_to_index = {label: idx for idx, label in enumerate(classes)}

    for label in classes:
        binary_true = (y_true == label).astype(int)
        idx = class_to_index[label]
        scores = y_proba[:, idx] 

        TPRs = []
        FPRs = []
        for threshold in thresholds:
            y_pred = (scores >= threshold).astype(int)
            TP, TN, FP, FN = get_trues_and_falses(binary_true, y_pred, label=1)
            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            TPRs.append(TPR)
            FPRs.append(FPR)

        roc_data[label] = (FPRs, TPRs)

    return roc_data

def AUC_ROC_multiclass(y_true, y_proba, thresholds):
    roc_data = curve_ROC_multiclass(y_true, y_proba, thresholds)
    aucs = []

    for label, (FPRs, TPRs) in roc_data.items():
        auc = np.trapz(TPRs, FPRs)
        aucs.append(auc)

    return np.mean(aucs)

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
    
def get_metrics_multiclass(y_true, y_pred, y_proba, thresholds_roc, thresholds_pr):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    y_proba = np.array(y_proba) 

    # metricas
    acc = accuracy(y_true, y_pred)
    prec = precision_multiclass(y_true, y_pred)
    rec = recall_multiclass(y_true, y_pred)
    f1 = f_score_multiclass(y_true, y_pred)
    auc_roc = AUC_ROC_multiclass(y_true, y_proba, thresholds_roc)
    auc_pr = AUC_PR_multiclass(y_true, y_proba, thresholds_pr)

    # mostrar tabla con pandas
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (Class 1)", "Precision (Class 2)", "Precision (Class 3)", 
                   "Recall (Class 1)", "Recall (Class 2)", "Recall (Class 3)", 
                   "F1-score (Class 1)", "F1-score (Class 2)", "F1-score (Class 3)", 
                   "AUC-ROC", "AUC-PR"],
        "Value": [acc, prec[0], prec[1], prec[2], 
                  rec[0], rec[1], rec[2], 
                  f1[0], f1[1], f1[2], 
                  auc_roc, auc_pr]
    })
    print("===== MÉTRICAS =====")
    print(metrics_df.to_string(index=False))

    # calcular curvas ROC
    roc_data = curve_ROC_multiclass(y_true, y_proba, thresholds_roc)

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
    plt.subplot(1, 3, 3)
    classes = np.unique(y_true)
    for i, label in enumerate(classes):
        precisions, recalls = curve_precision_recall(y_true, y_proba[:, i], thresholds_pr, label)
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

    return metrics_df

def get_numeric_metrics_multiclass(y_true_, y_pred_, y_proba, thresholds_roc, thresholds_pr):
    y_true = np.ravel(y_true_).copy()
    y_pred = np.ravel(y_pred_).copy()

    acc = accuracy(y_true, y_pred)
    pr = precision_multiclass(y_true, y_pred)
    rec = recall_multiclass(y_true, y_pred)
    f1 = f_score_multiclass(y_true, y_pred)
    auc_roc = AUC_ROC_multiclass(y_true, y_proba, thresholds_roc)
    auc_pr = AUC_PR_multiclass(y_true, y_proba, thresholds_pr)

    return [acc, pr, rec, f1, auc_roc, auc_pr]


def graph_all_for_3(metrics1, metrics2, metrics3, y_true_, y_proba1_, y_pred1_, y_proba2_, y_pred2_, y_proba3_, y_pred3_, thresholds_roc1, thresholds_pr1, thresholds_roc2, thresholds_pr2, thresholds_roc3, thresholds_pr3):
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", 
                   "Precision (Class 1)", "Precision (Class 2)", "Precision (Class 3)", 
                   "Recall (Class 1)", "Recall (Class 2)", "Recall (Class 3)", 
                   "F1-score (Class 1)", "F1-score (Class 2)", "F1-score (Class 3)", 
                   "AUC-ROC", "AUC-PR"],
        "LDA": [metrics1[0], metrics1[1][0], metrics1[1][1], metrics1[1][2], 
                    metrics1[2][0], metrics1[2][1], metrics1[2][2], 
                    metrics1[3][0], metrics1[3][1], metrics1[3][2], 
                    metrics1[4], metrics1[5]],
        "Random Forest": [metrics2[0], metrics2[1][0], metrics2[1][1], metrics2[1][2], 
                    metrics2[2][0], metrics2[2][1], metrics2[2][2], 
                    metrics2[3][0], metrics2[3][1], metrics2[3][2], 
                    metrics2[4], metrics2[5]],
        "Logistic Regression": [metrics3[0], metrics3[1][0], metrics3[1][1], metrics3[1][2], 
                    metrics3[2][0], metrics3[2][1], metrics3[2][2], 
                    metrics3[3][0], metrics3[3][1], metrics3[3][2], 
                    metrics3[4], metrics3[5]]
    })

    print("===== MÉTRICAS DE LOS TRES MODELOS =====")
    print(metrics_df.to_string(index=False))

    y_true = np.ravel(y_true_).copy()
    y_pred1 = np.ravel(y_pred1_).copy()
    y_pred2 = np.ravel(y_pred2_).copy()
    y_pred3 = np.ravel(y_pred3_).copy()
    y_proba1 = np.array(y_proba1_).copy()
    y_proba2 = np.array(y_proba2_).copy()
    y_proba3 = np.array(y_proba3_).copy()

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 3, 1)
    print("y true: ", y_true.shape)
    print("y pred: ", y_pred1.shape)
    matriz1 = confusion_matrix_multiclass(y_true, y_pred1)
    print("y true: ", y_true.shape)
    print("y pred: ", y_pred1.shape)
    classes1 = np.unique(np.concatenate((y_true, y_pred1)))
    df_cm1 = pd.DataFrame(matriz1, index=classes1, columns=classes1)
    sns.heatmap(df_cm1, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.title('Confusion Matrix - LDA')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.subplot(1, 3, 2)
    matriz2 = confusion_matrix_multiclass(y_true, y_pred2)
    classes2 = np.unique(np.concatenate((y_true, y_pred2)))
    df_cm2 = pd.DataFrame(matriz2, index=classes2, columns=classes2)
    sns.heatmap(df_cm2, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.subplot(1, 3, 3)
    matriz3 = confusion_matrix_multiclass(y_true, y_pred3)
    classes3 = np.unique(np.concatenate((y_true, y_pred3)))
    df_cm3 = pd.DataFrame(matriz3, index=classes3, columns=classes3)
    sns.heatmap(df_cm3, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.title('Confusion Matrix - Logistic Regression')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.rcParams.update({
        "font.size": 20,
    })
    plt.tight_layout()
    plt.show()

    thresholds_roc = np.linspace(0, 1, 100)

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    model_names = ["LDA", "Random Forest", "Logistic Regression"]
    colors = ["blue", "green", "red"]

    for i, (proba, thresholds_pr, name, color) in enumerate(zip(
        [y_proba1, y_proba2, y_proba3],
        [thresholds_pr1, thresholds_pr2, thresholds_pr3],
        model_names, colors)):
        
        classes = np.unique(y_true)
        for j, label in enumerate(classes):
            precisions, recalls = curve_precision_recall(y_true, proba[:, j], thresholds_pr, label)
            plt.plot(recalls, precisions, label=f"{name} - Clase {label}", color=color, alpha=0.5 + 0.2 * j)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for proba, thresholds_roc, name, color in zip(
        [y_proba1, y_proba2, y_proba3],
        [thresholds_roc1, thresholds_roc2, thresholds_roc3],
        model_names, colors):
        
        roc_data = curve_ROC_multiclass(y_true, proba, thresholds_roc)
        for label, (FPRs, TPRs) in roc_data.items():
            plt.plot(FPRs, TPRs, label=f"{name} - Clase {label}", alpha=0.5)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.rcParams.update({
        "font.size": 20,
    })
    plt.tight_layout()
    plt.show()

    return metrics_df


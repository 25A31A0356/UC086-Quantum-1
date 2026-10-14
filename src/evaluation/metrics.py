"""
Cancer Clinical Prediction Metrics & Evaluation Module.
Computes Accuracy, Balanced Accuracy, Precision, Recall, F1, Macro F1, ROC-AUC, and PR-AUC.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

def compute_cancer_metrics(y_true, y_pred, y_prob=None, model_name="Model", runtime_sec=0.0):
    """
    Computes full set of evaluation metrics suitable for imbalanced cancer mutation classification.
    """
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    mf1  = f1_score(y_true, y_pred, average='macro')
    
    auc_roc = roc_auc_score(y_true, y_prob) if y_prob is not None else float('nan')
    auc_pr  = average_precision_score(y_true, y_prob) if y_prob is not None else float('nan')
    
    return {
        'Model': model_name,
        'Accuracy': acc,
        'Balanced Accuracy': bacc,
        'Precision': prec,
        'Recall (Sensitivity)': rec,
        'F1-Score': f1,
        'Macro F1': mf1,
        'ROC-AUC': auc_roc,
        'PR-AUC': auc_pr,
        'Runtime (s)': runtime_sec
    }

def format_benchmark_table(metrics_list):
    """
    Formats a list of metric dictionaries into a display dataframe.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            'Model Architecture': m['Model'],
            'Accuracy': f"{m['Accuracy']*100:.1f}%",
            'Balanced Acc': f"{m['Balanced Accuracy']*100:.1f}%",
            'Precision': f"{m['Precision']:.3f}",
            'Recall': f"{m['Recall (Sensitivity)']:.3f}",
            'F1-Score': f"{m['F1-Score']:.3f}",
            'Macro F1': f"{m['Macro F1']:.3f}",
            'ROC-AUC': f"{m['ROC-AUC']:.3f}",
            'Train Time (s)': f"{m['Runtime (s)']:.2f}"
        })
    return pd.DataFrame(rows)

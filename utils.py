import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import auc, roc_curve, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

def evaluate_classifier(y_true, y_pred, classes):
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    cr = classification_report(y_true, y_pred, labels=classes, target_names=classes, zero_division=1)

    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report':cr,
    }
    
    return metrics_dict

def plot_confusion_matrix(y_true, y_pred, classes, cmap="Blues"): 
    
    cm = confusion_matrix(y_true, y_pred)

    cm_fig = plt.figure(figsize=(8, 6))
    cm_title = f"Confusion Matrix"
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False,
                xticklabels=classes,
                yticklabels=classes,
                annot_kws={"fontsize": 8})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(cm_title)
    plt.show()


def plot_auc(y_true, y_pred, probs, classes): 

    # create and save visualization AUC
    y_true = label_binarize(y_true, classes=range(len(classes))) 
    probs = np.array(probs)
    
    # Micro-average AUC
    fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    auc_fig = plt.figure(figsize=(8, 6))
    auc_title = f'Receiver Operating Characteristic (ROC) Curve'
    plt.plot(fpr_micro, tpr_micro, color='#6495ED', lw=2, label=f'ROC curve (AUC = {roc_auc_micro:.2f})')
    plt.plot([0, 1], [0, 1], color='#AED6F1', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(auc_title)
    plt.legend(loc='lower right')
    plt.show()
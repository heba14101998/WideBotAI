import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import auc, roc_curve, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize

def evaluate_classifier(y_true, y_pred, classes):
    """
    Evaluates a multi-class classification model using various metrics.

    This function calculates and returns a dictionary of common evaluation metrics, including accuracy, precision, 
    recall, F1-score, and a detailed classification report.

    Args:
        y_true (array-like): The true labels of the data.
        y_pred (array-like): The predicted labels from the model.
        classes (list): A list of class labels.

    Returns:
        dict: A dictionary containing the following evaluation metrics:
            - accuracy: Overall accuracy of the model.
            - precision: Weighted average precision across all classes.
            - recall: Weighted average recall across all classes.
            - f1_score: Weighted average F1-score across all classes.
            - classification_report: A detailed classification report with precision, recall, F1-score, and support for each class.
    """
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
    """
    Plots a confusion matrix to visualize the performance of a multi-class classification model.

    This function creates a heatmap representation of the confusion matrix, providing insights into the model's ability
    to correctly classify instances and identify common misclassifications.

    Args:
        y_true (array-like): The true labels of the data.
        y_pred (array-like): The predicted labels from the model.
        classes (list): A list of class labels.
        cmap (str, optional): The colormap to use for the heatmap. Defaults to "Blues".

    Returns:
        None
    """
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
    """
    Plots the Receiver Operating Characteristic (ROC) curve and calculates the Area Under the Curve (AUC).

    This function visualizes the model's performance for multi-class classification by plotting the ROC curve and 
    calculating the micro-average AUC.

    Args:
        y_true (array-like): The true labels of the data.
        y_pred (array-like): The predicted labels from the model.
        probs (array-like): The predicted probabilities for each class.
        classes (list): A list of class labels.

    Returns:
        None
    """
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


def learning_curve(accuracy, i):
    """
    Plots the learning curve of a model's accuracy over batches.

    This function visualizes the model's accuracy progression during training.
    It creates a line plot showing the accuracy values for each batch within a specific epoch.

    Args:
        accuracy (list): A list of accuracy values for each batch in an epoch.
        i (int): The epoch number.

    Returns:
        None
    """
    plt.figure(figsize=(10,4))
    plt.plot(accuracy, '-',color=palette_color[0])

    # Label the plot.
    plt.title(f"Model Accuracy in epoch {i} over batchs")
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")

    plt.show()
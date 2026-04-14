from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


def evaluate_model(model, X, y, verbose=True):
    """
    Works with both sklearn and TensorFlow models.
    """
    raw = model.predict(X)
    if raw.ndim == 2:
        y_pred = np.argmax(raw, axis=1)   
    else:
        y_pred = raw                     

    y      = np.array(y).astype(int)
    y_pred = np.array(y_pred).astype(int)

    acc         = accuracy_score(y, y_pred)
    f1_micro    = f1_score(y, y_pred, average='micro')
    f1_macro    = f1_score(y, y_pred, average='macro')
    f1_weighted = f1_score(y, y_pred, average='weighted')

    metrics = {
        "Accuracy":    acc,
        "F1-Micro":    f1_micro,
        "F1-Macro":    f1_macro,
        "F1-Weighted": f1_weighted
    }

    if verbose:
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred))

    return metrics, y_pred


def plot_confusion_matrix(y_true, y_pred, classes=None, title="Confusion Matrix"):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=classes,
        cmap=plt.cm.Blues,
        normalize=None
    )
    plt.title(title)
    # plt.show()
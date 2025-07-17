import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
)
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union


def plot_and_save_confusion_matrix(
    pipeline: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_subjects: Optional[int] = None
) -> None:
    """
    Plot and save the confusion matrix for a fitted classifier pipeline.

    Parameters
    ----------
    pipeline : sklearn estimator
        Fitted classification pipeline or model.
    X_test : array-like
        Test features.
    y_test : array-like
        True test labels.
    n_subjects : int or None, optional
        Number of subjects to include in filename and title (default None).
    """
    disp = ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
    title = f"Confusion Matrix (# Subjects: {n_subjects})" if n_subjects else "Confusion Matrix"
    plt.title(title)
    plt.grid(False)

    output_dir = os.path.join("output_img", "confusion_matrix")
    os.makedirs(output_dir, exist_ok=True)

    filename = f"confusion_matrix_{n_subjects}S.png" if n_subjects else "confusion_matrix.png"
    filepath = os.path.join(output_dir, filename)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Confusion matrix saved as '{filepath}'")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model_name: str = "Random Forest",
    save_dir: str = "output_img",
    n_subjects: Optional[int] = None,
    save_filename: str = "precision_recall_curve.png"
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Plot and save the Precision-Recall curve with Average Precision (AP) score.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_scores : array-like
        Predicted scores or probabilities for the positive class.
    model_name : str, optional
        Model name for legend and filename (default "Random Forest").
    save_dir : str, optional
        Directory to save the plot (default "output_img").
    n_subjects : int or None, optional
        Number of subjects to include in filename and title (default None).
    save_filename : str, optional
        Base filename for saving the plot (default "precision_recall_curve.png").

    Returns
    -------
    precision, recall, average_precision : tuple
        Arrays for precision, recall and average precision score.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.2f})')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    title = f'Precision-Recall Curve (# Subjects: {n_subjects})' if n_subjects else 'Precision-Recall Curve'
    plt.title(title)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(alpha=0.3)

    output_dir = os.path.join(save_dir, "precision_recall")
    os.makedirs(output_dir, exist_ok=True)

    base_name = save_filename.rsplit('.', 1)[0]
    model_str = model_name.replace(' ', '_').lower()
    if n_subjects:
        filename = f"{base_name}_{model_str}_{n_subjects}S.png"
    else:
        filename = save_filename
    filepath = os.path.join(output_dir, filename)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Average Precision (AP) score: {ap:.3f}")
    print(f"✅ Precision-Recall curve saved as '{filepath}'")

    return precision, recall, ap


def save_metrics_log(metrics_dict: Dict[str, Any], model: Any, n_subjects: int) -> None:
    """
    Save evaluation metrics to a formatted log file.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of metric names and their values.
    model : object
        Trained model instance (used for naming the log file).
    n_subjects : int
        Number of subjects (used for naming the log file).
    """
    model_name = type(model).__name__
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{model_name}_{n_subjects}S.txt"
    filepath = os.path.join(log_dir, filename)

    def format_value(value: Any) -> str:
        if isinstance(value, (list, np.ndarray)):
            vals = np.array(value).tolist() if isinstance(value, np.ndarray) else value
            lines = []
            line = ""
            for v in vals:
                s = f"{v:.3f}" if isinstance(v, (float, int)) else str(v)
                if len(line) + len(s) + 2 > 60:
                    lines.append(line.rstrip(", "))
                    line = ""
                line += s + ", "
            if line:
                lines.append(line.rstrip(", "))
            return "[\n    " + "\n    ".join(lines) + "\n]"
        elif isinstance(value, (float, int)):
            return f"{value:.3f}"
        else:
            return str(value)

    with open(filepath, "w") as f:
        f.write(f"Metrics log for model {model_name} ({n_subjects} subjects)\n")
        f.write("=" * 40 + "\n")
        for metric_name, metric_value in metrics_dict.items():
            f.write(f"{metric_name}:\n{format_value(metric_value)}\n\n")

    print(f"✅ Metrics saved to '{filepath}'")


def plot_and_save_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    output_dir: str = "output_img",
    filename_prefix: str = "roc_curve",
    n_subjects: Optional[int] = None
) -> None:
    """
    Plot the ROC curve with AUC and save as a PNG file.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    y_scores : array-like
        Predicted probabilities or scores for the positive class.
    output_dir : str, optional
        Directory to save the image (default "output_img").
    filename_prefix : str, optional
        Prefix for the saved filename (default "roc_curve").
    n_subjects : int or None, optional
        Number of subjects to include in filename and title (default None).
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    output_dir = os.path.join(output_dir, "roc_curve")
    os.makedirs(output_dir, exist_ok=True)

    filename = filename_prefix
    if n_subjects:
        filename += f"_{n_subjects}S"
    filename += ".png"
    filepath = os.path.join(output_dir, filename)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#1f77b4', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random guess')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    title = f'Receiver Operating Characteristic (ROC) (# Subjects: {n_subjects})' if n_subjects else 'Receiver Operating Characteristic (ROC)'
    plt.title(title)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(filepath, dpi=300)
    plt.show()
    plt.close()

    print(f"✅ ROC curve saved as '{filepath}'")

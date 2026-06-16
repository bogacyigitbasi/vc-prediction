"""
Evaluation metrics for all models.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """
    Compute all classification metrics.

    Args:
        y_true: ground truth binary labels
        y_pred: predicted binary labels (after thresholding)
        y_prob: predicted probabilities (for AUC)
    """
    return {
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def print_results(model_name: str, metrics: dict[str, float]) -> None:
    """Pretty-print model results."""
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")


def build_comparison_table(
    results: dict[str, dict[str, float]],
) -> str:
    """Build a markdown comparison table from results dict."""
    header = "| Model | F1 | ROC-AUC | Precision | Recall | Accuracy |"
    sep = "|-------|-----|---------|-----------|--------|----------|"
    rows = [header, sep]

    for model_name, metrics in results.items():
        row = (
            f"| {model_name} | "
            f"{metrics['f1']:.4f} | "
            f"{metrics['roc_auc']:.4f} | "
            f"{metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | "
            f"{metrics['accuracy']:.4f} |"
        )
        rows.append(row)

    return "\n".join(rows)

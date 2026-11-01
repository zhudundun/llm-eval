"""Calibration metrics for comparing a judge's scores against ground truth."""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)


def compute_calibration(
    judge_scores: list[float],
    ground_truth: list[float],
) -> dict:
    """Compute calibration metrics for a judge relative to a reference.

    Parameters
    ----------
    judge_scores:
        Binary scores (0.0 or 1.0) produced by the judge being evaluated.
    ground_truth:
        Reference binary scores (0.0 or 1.0) — typically from a programmatic
        judge or human annotators.

    Returns
    -------
    dict with keys:
        - ``accuracy``    : float — fraction of scores that agree
        - ``precision``   : float — precision for the positive (1.0) class
        - ``recall``      : float — recall for the positive (1.0) class
        - ``f1``          : float — F1 score for the positive (1.0) class
        - ``cohen_kappa`` : float — Cohen's kappa inter-rater agreement

    Raises
    ------
    ValueError
        If the two lists have different lengths or are empty.
    """
    if len(judge_scores) != len(ground_truth):
        raise ValueError(
            f"judge_scores and ground_truth must have the same length "
            f"({len(judge_scores)} vs {len(ground_truth)})"
        )
    if not judge_scores:
        raise ValueError("Score lists must not be empty.")

    # sklearn expects integer-like labels; cast to int for compatibility
    y_pred = [int(s) for s in judge_scores]
    y_true = [int(s) for s in ground_truth]

    accuracy: float = float(accuracy_score(y_true, y_pred))

    precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    precision: float = float(precision_arr)
    recall: float = float(recall_arr)
    f1: float = float(f1_arr)

    kappa: float = float(cohen_kappa_score(y_true, y_pred))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cohen_kappa": kappa,
    }


def print_calibration_report(results: dict) -> None:
    """Pretty-print a calibration metrics table to stdout.

    Parameters
    ----------
    results:
        A dict as returned by :func:`compute_calibration`.
    """
    _KEY_LABELS = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "cohen_kappa": "Cohen's Kappa",
    }

    col_width = 20
    separator = "-" * (col_width * 2 + 3)

    print(separator)
    print(f"{'Metric':<{col_width}}  {'Value':>{col_width}}")
    print(separator)
    for key, label in _KEY_LABELS.items():
        value = results.get(key)
        if value is None:
            formatted = "N/A"
        else:
            formatted = f"{value:.4f}"
        print(f"{label:<{col_width}}  {formatted:>{col_width}}")
    print(separator)

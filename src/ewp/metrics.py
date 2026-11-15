"""Metrics suited to imbalanced early-warning problems."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def recall_at_capacity(y_true, scores, frac: float) -> float:
    """Share of true positives captured if only the top `frac` of cases are reviewed."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    k = max(1, int(round(frac * len(y_true))))
    top = np.argsort(-scores)[:k]
    total_pos = y_true.sum()
    return float(y_true[top].sum() / total_pos) if total_pos else float("nan")


def evaluate(y_true, scores, capacities=(0.05, 0.10, 0.20)) -> dict:
    out = {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "brier": brier_score_loss(y_true, scores),
        "base_rate": float(np.mean(y_true)),
    }
    for c in capacities:
        out[f"recall@{int(c * 100)}%"] = recall_at_capacity(y_true, scores, c)
    return out

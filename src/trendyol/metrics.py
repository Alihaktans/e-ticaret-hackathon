from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, classification_report


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def find_best_threshold(y_true, y_proba, start: float = 0.05, end: float = 0.95, steps: int = 91) -> dict:
    thresholds = np.linspace(start, end, steps)

    best = {
        "threshold": 0.5,
        "macro_f1": -1.0,
    }

    for threshold in thresholds:
        pred = (y_proba >= threshold).astype(int)
        score = macro_f1(y_true, pred)

        if score > best["macro_f1"]:
            best = {
                "threshold": float(threshold),
                "macro_f1": float(score),
            }

    return best


def make_classification_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, digits=5)

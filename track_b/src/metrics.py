"""metrics.py — Macro-F1, per-class F1, confusion, margin, entropy.

Dipakai oleh diagnose.py (verdict underfit/overfit) dan oof.py (validasi).
Label order terkunci: 0=Recyclable, 1=Electronic, 2=Organic.
"""
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

LABELS = [0, 1, 2]  # 0=Recyclable, 1=Electronic, 2=Organic


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0.0)
    )


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0.0)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def margin(probs: np.ndarray) -> np.ndarray:
    part = np.sort(probs, axis=1)
    return part[:, -1] - part[:, -2]


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)

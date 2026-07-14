"""consistency.py — Pengaman anti overfit-OOF.

Iterasi di atas embedding cache gratis (CPU, menit), dan itu bahaya barunya:
kalau mencoba 200 kombinasi lalu memilih yang tertinggi di OOF, hasilnya
overfit ke OOF dan skor test turun. Gain kecil yang konsisten di semua fold
> gain besar yang cuma muncul di satu fold -- fold_consistency() memberi
angka (`std`, `min`) untuk menegakkan aturan itu.
"""
import numpy as np

from metrics import macro_f1


def fold_consistency(oof: np.ndarray, folds_df) -> dict:
    y = folds_df["label"].to_numpy()
    pred = oof.argmax(axis=1)
    per_fold = [
        macro_f1(y[folds_df["fold"] == f], pred[folds_df["fold"] == f])
        for f in sorted(folds_df["fold"].unique())
    ]
    return {
        "per_fold": per_fold,
        "mean": float(np.mean(per_fold)),
        "std": float(np.std(per_fold)),
        "min": float(np.min(per_fold)),
    }

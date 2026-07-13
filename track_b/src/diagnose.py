"""diagnose.py — Verdict underfit/overfit + tabel action per kelas.

Kenapa tabel: reviewer menilai proses, bukan cuma satu angka CV. Tabel
"kelas mana yang lemah, tertukar dengan siapa, dan tindakan apa" jauh lebih
kuat di report. Action diambil dari confusion matrix asli, bukan hardcode.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from metrics import confusion

CLASS_NAMES = {0: "Recyclable", 1: "Electronic", 2: "Organic"}


def verdict(train_f1: float, val_f1: float,
            gap_threshold: float = 0.05, underfit_ceiling: float = 0.97) -> str:
    """Train F1 belum mentok -> underfit. Train mentok tapi val jauh -> overfit."""
    if train_f1 < underfit_ceiling:
        return "underfit"
    if (train_f1 - val_f1) > gap_threshold:
        return "overfit"
    return "ok"


def class_action_table(y_true, y_pred, target_f1: float = 0.99) -> pd.DataFrame:
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0.0
    )
    cm = confusion(y_true, y_pred)

    rows = []
    for c in [0, 1, 2]:
        off_diag = cm[c].copy()
        off_diag[c] = -1  # abaikan prediksi yang benar
        partner = int(np.argmax(off_diag))
        n_confused = int(cm[c][partner])
        gap = max(0.0, target_f1 - float(f1[c]))

        if gap == 0:
            action = "Sudah di target; jangan diutak-atik."
        elif r[c] < p[c]:
            action = (f"Recall rendah (banyak {CLASS_NAMES[c]} lolos jadi "
                      f"{CLASS_NAMES[partner]}) -> turunkan threshold kelas ini di Track C; "
                      f"cek kandidat mislabel pasangan {CLASS_NAMES[c]}<->{CLASS_NAMES[partner]}.")
        else:
            action = (f"Precision rendah (banyak {CLASS_NAMES[partner]} salah masuk "
                      f"{CLASS_NAMES[c]}) -> naikkan threshold kelas ini; "
                      f"review sampel ambigu pasangan ini.")

        rows.append({
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "precision": float(p[c]),
            "recall": float(r[c]),
            "f1": float(f1[c]),
            "support": int(s[c]),
            "gap_to_target": gap,
            "top_confusion_with": CLASS_NAMES[partner],
            "n_confused": n_confused,
            "recommended_action": action,
        })

    return pd.DataFrame(rows)

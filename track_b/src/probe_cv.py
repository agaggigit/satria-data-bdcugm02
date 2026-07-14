"""probe_cv.py — 5-fold CV di atas embedding beku -> OOF leak-proof.

Reuse assemble_oof/validate_oof dari oof.py apa adanya (bukan ditulis ulang) --
modul itu sudah mengunci bug alignment/kebocoran OOF yang sama jenisnya di
sini: model tidak boleh pernah melihat baris yang divalidasinya.
"""
import numpy as np

from heads import make_head
from metrics import macro_f1
from oof import assemble_oof, validate_oof


def run_probe_cv(X: np.ndarray, folds_df, head_name: str,
                 class_weight=None, seed: int = 42) -> tuple:
    assert X.shape[0] == len(folds_df), "X tidak align dengan folds_df"
    y = folds_df["label"].to_numpy()

    fold_probs, scores = {}, []
    for f in sorted(folds_df["fold"].unique()):
        tr = folds_df.index[folds_df["fold"] != f].to_numpy()
        va = folds_df.index[folds_df["fold"] == f].to_numpy()

        head = make_head(head_name, seed=seed, class_weight=class_weight)
        head.fit(X[tr], y[tr])
        probs = head.predict_proba(X[va])

        fold_probs[int(f)] = (va, probs)
        scores.append(macro_f1(y[va], probs.argmax(axis=1)))

    oof = assemble_oof(fold_probs, n_rows=len(folds_df))
    validate_oof(oof, folds_df)
    return oof, scores

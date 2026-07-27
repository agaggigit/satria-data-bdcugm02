"""export_misclassified.py — Ekstrak Sampel-sampel Salah Tebak (0.99% Error Rate)
Model: SigLIP2-SO400M + kNN (k=15, Cosine) di data folds.csv (v1)

Menghasilkan file CSV daftar sampel yang salah ditebak beserta:
- Filepath / filename gambar
- True Label vs Predicted Label
- Probability per-kelas (prob_0, prob_1, prob_2) & Confidence score
"""
import os
import sys
import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

from embed import load_embeddings
from features import l2norm
from heads import make_head
from oof import assemble_oof, validate_oof
from diagnose import CLASS_NAMES


def extract_misclassified_df(X: np.ndarray, folds_df: pd.DataFrame,
                              head_name: str = "knn", seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jalankan OOF 5-fold dan ekstrak DataFrame sampel yang salah ditebak."""
    y_true = folds_df["label"].to_numpy()
    folds = folds_df["fold"].to_numpy()
    unique_folds = sorted(folds_df["fold"].unique())

    fold_probs = {}
    for f in unique_folds:
        tr_mask = folds != f
        va_mask = folds == f

        head = make_head(head_name, seed=seed)
        head.fit(X[tr_mask], y_true[tr_mask])
        probs = head.predict_proba(X[va_mask])
        val_idx = folds_df.index[va_mask].to_numpy()
        fold_probs[int(f)] = (val_idx, probs)

    oof_probs = assemble_oof(fold_probs, n_rows=len(folds_df))
    validate_oof(oof_probs, folds_df)

    y_pred = oof_probs.argmax(axis=1)
    wrong_mask = y_pred != y_true

    wrong_df = folds_df[wrong_mask].copy()
    wrong_probs = oof_probs[wrong_mask]
    wrong_preds = y_pred[wrong_mask]

    wrong_df["true_label_name"] = wrong_df["label"].map(CLASS_NAMES)
    wrong_df["pred_label_id"] = wrong_preds
    wrong_df["pred_label_name"] = wrong_df["pred_label_id"].map(CLASS_NAMES)
    wrong_df["prob_recyclable"] = wrong_probs[:, 0]
    wrong_df["prob_electronic"] = wrong_probs[:, 1]
    wrong_df["prob_organic"] = wrong_probs[:, 2]
    wrong_df["confidence"] = wrong_probs.max(axis=1)

    # Tabel ringkasan pasangan tertukar
    summary_list = []
    for (t, p), group in wrong_df.groupby(["true_label_name", "pred_label_name"]):
        summary_list.append({
            "true_class": t,
            "predicted_as": p,
            "n_samples": len(group),
            "pct_of_all_errors": f"{100 * len(group) / len(wrong_df):.2f}%"
        })
    summary_df = pd.DataFrame(summary_list).sort_values(by="n_samples", ascending=False)

    return wrong_df, summary_df


def main(use_v2: bool = True):
    from config import CFG
    from embed import align_to_subset

    folds_v1 = pd.read_csv(CFG.folds_csv)
    target_csv = CFG.folds_v2_csv if use_v2 else CFG.folds_csv
    dataset_name = "folds_v2.csv (Data Bersih Track A)" if use_v2 else "folds.csv (Data Asli v1)"

    print(f"--- Loading Embeddings SigLIP2-SO400M & Dataset: {dataset_name} ---")
    folds_df = pd.read_csv(target_csv)
    X_full = l2norm(load_embeddings("siglip2so400m", "train")[0])
    
    if use_v2:
        X = align_to_subset(X_full, folds_v1, folds_df)
    else:
        X = X_full

    print("--- Mengekstraksi Sampel Misclassified (SO400M + kNN) ---")
    wrong_df, summary_df = extract_misclassified_df(X, folds_df, head_name="knn", seed=CFG.seed)

    error_rate = 100 * len(wrong_df) / len(folds_df)
    accuracy = 100 - error_rate

    print(f"\n=======================================================")
    print(f"📊 HASIL EVALUASI DIAGNOSIS MISLABEL [{dataset_name}]")
    print(f"=======================================================")
    print(f" Total Sampel Dataset : {len(folds_df):,} gambar")
    print(f" Akurasi Model (kNN)  : {accuracy:.2f}%")
    print(f" Jumlah Mislabel/Salah: {len(wrong_df):,} gambar ({error_rate:.2f}% Error Rate)")
    print(f"=======================================================")
    print("\n=== RINGKASAN PASANGAN TERTUKAR (TOP CONFUSION) ===")
    print(summary_df.to_string(index=False))

    suffix = "_v2" if use_v2 else "_v1"
    out_csv = os.path.join(os.path.dirname(__file__), "..", "results", f"misclassified_so400m_knn{suffix}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    wrong_df.to_csv(out_csv, index=False)
    print(f"\n📄 Daftar lengkap {len(wrong_df)} sampel salah tersimpan di: {out_csv}")


if __name__ == "__main__":
    main()

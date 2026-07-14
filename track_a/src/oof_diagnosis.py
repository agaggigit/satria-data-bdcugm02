"""
oof_diagnosis.py — Fase 0: Diagnosis OOF Track B
Track A — BDC Satria Data 2026

Input : oof.npy [N, 3] + folds.csv
Output: per-class F1, confusion matrix, distribusi margin & entropy,
        error rate per fold, DataFrame diagnosis siap untuk Fase 1.

Jalankan via notebook 03_oof_diagnosis.ipynb di Colab.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
)


# ─── Konstanta ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Recyclable", "Electronic", "Organic"]


# ─── Step 1: Validasi OOF ─────────────────────────────────────────────────────

def validate_oof(oof: np.ndarray, folds_df: pd.DataFrame) -> None:
    """
    Pastikan oof.npy valid sebelum dipakai untuk apapun.
    Raise AssertionError jika ada masalah.
    """
    n = len(folds_df)
    assert oof.ndim == 2 and oof.shape[1] == 3, \
        f"Shape OOF harus [N, 3], dapat: {oof.shape}"
    assert oof.shape[0] == n, \
        f"Jumlah baris OOF ({oof.shape[0]}) != folds.csv ({n})"
    n_missing = int(np.isnan(oof).any(axis=1).sum())
    assert n_missing == 0, f"{n_missing} baris OOF mengandung NaN"
    sums = oof.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-3), \
        f"Probabilitas OOF tidak sum ke 1 (min={sums.min():.4f}, max={sums.max():.4f})"
    assert set(folds_df["label"].unique()) <= {0, 1, 2}, \
        f"Label di luar 0/1/2: {folds_df['label'].unique()}"
    print(f"[validate_oof] ✅ OOF valid — shape {oof.shape}, "
          f"prob sum range [{sums.min():.5f}, {sums.max():.5f}]")


# ─── Step 2: Per-class F1 + Confusion Matrix ─────────────────────────────────

def compute_per_class_metrics(
    oof: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Hitung argmax prediction, per-class F1, dan confusion matrix.

    Returns dict berisi:
        preds          : np.ndarray [N] — argmax predictions
        f1_per_class   : list[float] — F1 per kelas [Recyclable, Electronic, Organic]
        f1_macro       : float — Macro-F1 keseluruhan
        conf_matrix    : np.ndarray [3, 3] — confusion matrix
        report_str     : str — classification report lengkap
    """
    preds = oof.argmax(axis=1)

    f1_per_class = f1_score(labels, preds, average=None, labels=[0, 1, 2])
    f1_macro = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)

    print("\n" + "=" * 60)
    print("DIAGNOSIS: Per-class F1 + Confusion Matrix")
    print("=" * 60)
    for i, (name, f1) in enumerate(zip(CLASS_NAMES, f1_per_class)):
        flag = " ⚠️ TERENDAH" if f1 == f1_per_class.min() else ""
        print(f"  {name:15s}: F1 = {f1:.4f}{flag}")
    print(f"\n  Macro-F1 OOF  : {f1_macro:.4f}")
    print(f"\n  Classification Report:\n{report}")

    # Konfirmasi klaim reviewer
    recyclable_f1 = f1_per_class[0]
    organic_f1 = f1_per_class[2]
    electronic_f1 = f1_per_class[1]
    if recyclable_f1 < electronic_f1 and organic_f1 < electronic_f1:
        print("  ✅ KONFIRMASI REVIEWER: Recyclable & Organic memang F1 terendah")
    elif recyclable_f1 < electronic_f1 or organic_f1 < electronic_f1:
        print("  ⚠️ PARSIAL: Salah satu dari Recyclable/Organic lebih rendah dari Electronic")
    else:
        print("  ❌ TIDAK TERKONFIRMASI: Electronic justru yang terendah")

    return {
        "preds": preds,
        "f1_per_class": f1_per_class.tolist(),
        "f1_macro": f1_macro,
        "conf_matrix": cm,
        "report_str": report,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (7, 6),
) -> None:
    """Plot confusion matrix yang rapi dengan normalisasi baris."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, mat, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (Count)", "Confusion Matrix (Row-Normalized)"],
        ["d", ".2%"],
    ):
        sns.heatmap(
            mat, annot=True, fmt=fmt, ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cmap="Blues", linewidths=0.5,
        )
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)

    plt.suptitle("OOF Confusion Matrix — BDC 2026", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_confusion_matrix] Disimpan ke {save_path}")
    plt.show()


# ─── Step 3: Margin & Entropy ─────────────────────────────────────────────────

def compute_margin_entropy(oof: np.ndarray) -> pd.DataFrame:
    """
    Hitung margin (top1 - top2) dan entropy per sampel.

    Returns DataFrame dengan kolom:
        top1_prob, top2_prob, margin, entropy, pred_class
    """
    sorted_probs = np.sort(oof, axis=1)[:, ::-1]   # descending
    top1 = sorted_probs[:, 0]
    top2 = sorted_probs[:, 1]
    margin = top1 - top2
    ent = scipy_entropy(oof.T)  # entropy per baris, shape [N]

    df = pd.DataFrame({
        "top1_prob": top1,
        "top2_prob": top2,
        "margin":    margin,
        "entropy":   ent,
        "pred_class": oof.argmax(axis=1),
    })

    print(f"\n[compute_margin_entropy] Distribusi margin:")
    print(f"  Min    : {margin.min():.4f}")
    print(f"  P5     : {np.percentile(margin, 5):.4f}")
    print(f"  Median : {np.median(margin):.4f}")
    print(f"  P95    : {np.percentile(margin, 95):.4f}")
    print(f"  Max    : {margin.max():.4f}")
    print(f"\n[compute_margin_entropy] Distribusi entropy:")
    print(f"  Min    : {ent.min():.4f}")
    print(f"  P95    : {np.percentile(ent, 95):.4f}")
    print(f"  Max    : {ent.max():.4f}")

    return df


def plot_margin_entropy_dist(
    margin_df: pd.DataFrame,
    labels: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot distribusi margin dan entropy per kelas."""
    df = margin_df.copy()
    df["true_label"] = labels
    df["class_name"] = df["true_label"].map(dict(enumerate(CLASS_NAMES)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, title in zip(
        axes,
        ["margin", "entropy"],
        ["Distribusi Margin (top1 − top2) per Kelas",
         "Distribusi Entropy per Kelas"],
    ):
        for i, name in enumerate(CLASS_NAMES):
            subset = df[df["true_label"] == i][col]
            ax.hist(subset, bins=60, alpha=0.6, label=name, density=True)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(col.capitalize(), fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Analisis Ketidakpastian OOF — Track A Cleaning", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_margin_entropy_dist] Disimpan ke {save_path}")
    plt.show()


# ─── Step 4: Error Rate per Fold ─────────────────────────────────────────────

def compute_fold_error_rates(
    oof: np.ndarray,
    folds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Hitung error rate (1 - Macro-F1) per fold untuk deteksi fold anomali.

    Returns DataFrame dengan kolom: fold, n_val, macro_f1, error_rate
    """
    rows = []
    preds = oof.argmax(axis=1)
    for fold_id in sorted(folds_df["fold"].unique()):
        mask = folds_df["fold"] == fold_id
        y_true = folds_df.loc[mask, "label"].values
        y_pred = preds[mask]
        f1 = f1_score(y_true, y_pred, average="macro")
        rows.append({
            "fold":       fold_id,
            "n_val":      mask.sum(),
            "macro_f1":   round(f1, 4),
            "error_rate": round(1 - f1, 4),
        })

    df = pd.DataFrame(rows)
    print("\n[compute_fold_error_rates] Error rate per fold:")
    print(df.to_string(index=False))

    max_fold = df.loc[df["error_rate"].idxmax()]
    min_fold = df.loc[df["error_rate"].idxmin()]
    diff = max_fold["error_rate"] - min_fold["error_rate"]
    if diff > 0.02:
        print(f"\n  ⚠️ Fold {max_fold['fold']} error rate jauh lebih tinggi "
              f"(selisih {diff:.3f}) — periksa kemungkinan distribusi aneh di fold ini")
    else:
        print(f"\n  ✅ Error rate antar fold konsisten (range: {diff:.3f})")

    return df


# ─── Step 5: Build Diagnosis DataFrame ───────────────────────────────────────

def build_diagnosis_df(
    oof: np.ndarray,
    folds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gabungkan semua informasi diagnosis ke dalam satu DataFrame.
    Ini yang jadi input untuk Fase 1 (cleanlab) dan Fase 2 (ambiguous filter).

    Kolom output:
        filepath, label, fold, pred_class, is_correct,
        top1_prob, top2_prob, margin, entropy
    """
    margin_df = compute_margin_entropy(oof)
    preds = oof.argmax(axis=1)

    diag = folds_df[["filepath", "label", "fold"]].copy().reset_index(drop=True)
    diag["pred_class"]  = preds
    diag["is_correct"]  = (diag["label"] == diag["pred_class"]).astype(int)
    diag["top1_prob"]   = margin_df["top1_prob"].values
    diag["top2_prob"]   = margin_df["top2_prob"].values
    diag["margin"]      = margin_df["margin"].values
    diag["entropy"]     = margin_df["entropy"].values
    diag["oof_prob_0"]  = oof[:, 0]
    diag["oof_prob_1"]  = oof[:, 1]
    diag["oof_prob_2"]  = oof[:, 2]

    n_error = (diag["is_correct"] == 0).sum()
    print(f"\n[build_diagnosis_df] Total sampel   : {len(diag):,}")
    print(f"[build_diagnosis_df] Salah prediksi  : {n_error:,} ({n_error/len(diag)*100:.1f}%)")

    return diag


# ─── Main: jalankan semua diagnosis sekaligus ─────────────────────────────────

def run_full_diagnosis(
    oof_path: str,
    folds_csv: str,
    output_dir: Optional[str] = None,
    save_plots: bool = True,
) -> dict:
    """
    Entry point utama. Jalankan seluruh pipeline diagnosis OOF.

    Args:
        oof_path   : path ke oof.npy [N, 3]
        folds_csv  : path ke folds.csv
        output_dir : folder untuk simpan plot & CSV (opsional)
        save_plots : simpan plot ke file

    Returns:
        dict berisi semua hasil diagnosis
    """
    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FASE 0: OOF DIAGNOSIS — BDC Satria Data 2026")
    print("=" * 60)

    # Load
    oof = np.load(oof_path)
    folds_df = pd.read_csv(folds_csv)
    labels = folds_df["label"].values

    # Step 1 — Validasi
    validate_oof(oof, folds_df)

    # Step 2 — Metrics
    metrics = compute_per_class_metrics(oof, labels)

    # Step 3 — Margin & Entropy
    margin_df = compute_margin_entropy(oof)

    # Step 4 — Fold error rates
    fold_df = compute_fold_error_rates(oof, folds_df)

    # Step 5 — Diagnosis DataFrame
    diag_df = build_diagnosis_df(oof, folds_df)

    # Plot
    if save_plots and out_dir:
        plot_confusion_matrix(
            metrics["conf_matrix"],
            save_path=str(out_dir / "oof_confusion_matrix.png"),
        )
        plot_margin_entropy_dist(
            diag_df,
            labels=labels,
            save_path=str(out_dir / "oof_margin_entropy_dist.png"),
        )
    else:
        plot_confusion_matrix(metrics["conf_matrix"])
        plot_margin_entropy_dist(diag_df, labels=labels)

    # Save diagnosis CSV
    if out_dir:
        diag_path = out_dir / "oof_diagnosis.csv"
        diag_df.to_csv(diag_path, index=False)
        print(f"\n[run_full_diagnosis] Disimpan: {diag_path}")

    print("\n✅ Fase 0 selesai!")

    return {
        "oof":         oof,
        "folds_df":    folds_df,
        "labels":      labels,
        "metrics":     metrics,
        "diag_df":     diag_df,
        "fold_df":     fold_df,
    }

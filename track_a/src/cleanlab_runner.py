"""
cleanlab_runner.py — Fase 1: Confident Learning dengan Cleanlab
Track A — BDC Satria Data 2026

Input : oof.npy [N, 3] + diagnosis DataFrame (dari oof_diagnosis.py)
Output: label_issues.csv — daftar kandidat mislabel dengan ranking confidence

Catatan penting:
- WAJIB pakai prediksi out-of-fold (bukan in-sample)
- Relabel > drop untuk kasus mislabel yang jelas
- Export ke label_issues.csv siap untuk review visual (contact_sheet.py)

Jalankan via notebook 04_cleanlab_cleaning.ipynb di Colab.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Cek cleanlab tersedia
try:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False
    print("[cleanlab_runner] ⚠️ cleanlab tidak terinstall.")
    print("  Install: pip install cleanlab")


# ─── Konstanta ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Recyclable", "Electronic", "Organic"]
CLASS_PAIRS_REVIEW_PRIORITY = [
    (0, 2, "Recyclable→Organic"),   # pasangan yang paling sering tertukar (sesuai reviewer)
    (2, 0, "Organic→Recyclable"),
    (0, 1, "Recyclable→Electronic"),
    (1, 0, "Electronic→Recyclable"),
    (1, 2, "Electronic→Organic"),
    (2, 1, "Organic→Electronic"),
]


# ─── Step 1: Jalankan Cleanlab ────────────────────────────────────────────────

def run_cleanlab(
    oof: np.ndarray,
    labels: np.ndarray,
    filter_by: str = "prune_by_noise_rate",
    frac_noise: float = 0.1,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Jalankan cleanlab find_label_issues() pada seluruh dataset.

    Args:
        oof        : probabilitas OOF [N, 3] — HARUS out-of-fold
        labels     : label asli [N] — integer 0/1/2
        filter_by  : metode cleanlab ('prune_by_noise_rate' atau 'both')
        frac_noise : fraksi noise yang diasumsikan (default 0.1 = 10%)
        n_jobs     : paralel jobs (gunakan 1 di Colab untuk stabilitas)

    Returns:
        DataFrame issues dengan kolom:
            sample_idx, is_label_issue, label_quality_score,
            label_asli, label_predicted, confidence
    """
    if not CLEANLAB_AVAILABLE:
        raise ImportError("cleanlab tidak tersedia. pip install cleanlab")

    print("[run_cleanlab] Menjalankan cleanlab find_label_issues()...")
    print(f"  Filter method : {filter_by}")
    print(f"  frac_noise    : {frac_noise}")
    print(f"  Dataset size  : {len(labels):,} sampel")

    # find_label_issues mengembalikan boolean array [N]
    label_issues_bool = find_label_issues(
        labels=labels,
        pred_probs=oof.astype(np.float32),
        filter_by=filter_by,
        frac_noise=frac_noise,
        n_jobs=n_jobs,
        return_indices_ranked_by="self_confidence",
    )

    # Label quality score per sampel
    quality_scores = get_label_quality_scores(
        labels=labels,
        pred_probs=oof.astype(np.float32),
    )

    # Prediksi argmax
    pred_labels = oof.argmax(axis=1)

    # Build DataFrame
    issues_df = pd.DataFrame({
        "sample_idx":          np.where(label_issues_bool)[0]
                               if isinstance(label_issues_bool, np.ndarray)
                               else label_issues_bool,
    })

    # Kalau find_label_issues return indices (cleanlab >= 2.x)
    if label_issues_bool.dtype == int or label_issues_bool.dtype == np.intp:
        issue_idx = label_issues_bool
    else:
        # boolean mask
        issue_idx = np.where(label_issues_bool)[0]

    n_issues = len(issue_idx)
    print(f"\n[run_cleanlab] Kandidat label issues: {n_issues:,} "
          f"({n_issues/len(labels)*100:.2f}% dari dataset)")

    result_df = pd.DataFrame({
        "sample_idx":         issue_idx,
        "label_quality_score": quality_scores[issue_idx],
        "label_asli":          labels[issue_idx],
        "label_predicted":     pred_labels[issue_idx],
        "confidence":          oof[issue_idx].max(axis=1),
        "margin":              (oof[issue_idx, pred_labels[issue_idx]] -
                                np.partition(oof[issue_idx], -2, axis=1)[:, -2]),
    })

    # Tambah nama kelas
    result_df["label_asli_name"]      = result_df["label_asli"].map(dict(enumerate(CLASS_NAMES)))
    result_df["label_predicted_name"] = result_df["label_predicted"].map(dict(enumerate(CLASS_NAMES)))

    # Flag double-flagged (mislabel + margin rendah = paling mencurigakan)
    margin_threshold = np.percentile(result_df["label_quality_score"], 25)
    result_df["is_high_priority"] = result_df["label_quality_score"] <= margin_threshold

    # Urutkan dari paling mencurigakan (quality score terendah)
    result_df = result_df.sort_values("label_quality_score").reset_index(drop=True)

    print(f"  High-priority  : {result_df['is_high_priority'].sum():,} sampel")

    # Distribusi per pasangan kelas
    print("\n  Distribusi per pasangan kelas (label_asli → label_predicted):")
    pair_counts = result_df.groupby(
        ["label_asli_name", "label_predicted_name"]
    ).size().reset_index(name="count").sort_values("count", ascending=False)
    print(pair_counts.to_string(index=False))

    return result_df


# ─── Step 2: Gabung dengan filepath dari folds.csv ───────────────────────────

def merge_with_filepaths(
    issues_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    diag_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Gabungkan result cleanlab dengan filepath dari folds.csv.
    Opsional: juga merge dengan diagnosis df (margin, entropy dari Fase 0).

    Returns:
        DataFrame dengan kolom:
            filepath, label_asli, label_predicted, label_quality_score,
            confidence, margin, entropy (jika diag_df tersedia),
            label_asli_name, label_predicted_name, is_high_priority
    """
    merged = issues_df.copy()
    merged["filepath"] = folds_df.iloc[merged["sample_idx"].values]["filepath"].values
    merged["fold"]     = folds_df.iloc[merged["sample_idx"].values]["fold"].values

    if diag_df is not None:
        # Tambah entropy dari diagnosis
        entropy_map = diag_df["entropy"].values
        merged["entropy"] = entropy_map[merged["sample_idx"].values]

    return merged


# ─── Step 3: Export label_issues.csv ─────────────────────────────────────────

def export_label_issues(
    merged_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Export kandidat label issues ke CSV untuk arsip dan review.
    Kolom: filepath, label_asli, label_predicted, label_quality_score,
           confidence, margin, entropy, label_asli_name, label_predicted_name,
           is_high_priority, fold
    """
    cols_order = [
        "filepath", "fold",
        "label_asli", "label_asli_name",
        "label_predicted", "label_predicted_name",
        "label_quality_score", "confidence", "margin",
        "is_high_priority",
    ]
    if "entropy" in merged_df.columns:
        cols_order.insert(-1, "entropy")

    export_cols = [c for c in cols_order if c in merged_df.columns]
    merged_df[export_cols].to_csv(output_path, index=False)
    print(f"[export_label_issues] Disimpan ke {output_path}")
    print(f"  {len(merged_df):,} kandidat | "
          f"{merged_df['is_high_priority'].sum():,} high-priority")


# ─── Fallback: tanpa cleanlab (dari margin threshold saja) ───────────────────

def run_margin_based_issues(
    diag_df: pd.DataFrame,
    margin_threshold: float = 0.15,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Alternatif jika cleanlab tidak tersedia atau sebagai cross-check.
    Identifikasi kandidat mislabel berdasarkan:
    1. Prediksi salah (is_correct == 0) AND
    2. Margin rendah (model yakin dengan kelas yang salah = mislabel jelas)
       ATAU margin sangat rendah (model tidak yakin = ambigu)

    Args:
        diag_df          : output build_diagnosis_df() dari oof_diagnosis.py
        margin_threshold : threshold margin — di bawah ini = kandidat (default 0.15)
        top_n            : ambil N teratas saja (None = semua)

    Returns:
        DataFrame kandidat mislabel berdasarkan margin
    """
    # Sampel yang salah prediksi dengan margin rendah = mislabel paling meyakinkan
    wrong_low_margin = diag_df[
        (diag_df["is_correct"] == 0) & (diag_df["margin"] < margin_threshold)
    ].copy()

    wrong_low_margin["source"] = "wrong_prediction_low_margin"
    wrong_low_margin["is_high_priority"] = wrong_low_margin["margin"] < (margin_threshold / 2)

    result = wrong_low_margin.sort_values("margin").reset_index(drop=True)

    if top_n:
        result = result.head(top_n)

    print(f"[run_margin_based_issues] Kandidat (margin < {margin_threshold}): {len(result):,}")
    print(f"  High-priority (margin < {margin_threshold/2:.3f}): "
          f"{result['is_high_priority'].sum():,}")

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_full_cleanlab(
    oof: np.ndarray,
    folds_df: pd.DataFrame,
    diag_df: Optional[pd.DataFrame],
    output_dir: str,
    use_cleanlab: bool = True,
    margin_threshold: float = 0.15,
    frac_noise: float = 0.10,
) -> pd.DataFrame:
    """
    Entry point Fase 1. Jalankan cleanlab (atau fallback margin-based) dan
    export hasil ke label_issues.csv.

    Returns:
        DataFrame label issues (sudah di-merge dengan filepath)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = folds_df["label"].values

    if use_cleanlab and CLEANLAB_AVAILABLE:
        issues_df = run_cleanlab(oof, labels, frac_noise=frac_noise)
        merged = merge_with_filepaths(issues_df, folds_df, diag_df)
        method = "cleanlab"
    else:
        print("[run_full_cleanlab] Menggunakan fallback margin-based issues")
        if diag_df is None:
            raise ValueError("diag_df wajib ada untuk margin-based fallback")
        merged = run_margin_based_issues(diag_df, margin_threshold=margin_threshold)
        method = "margin_based"

    # Export
    out_path = out_dir / "label_issues.csv"
    export_label_issues(merged, str(out_path))
    print(f"\n✅ Fase 1 selesai! Method: {method}")

    return merged

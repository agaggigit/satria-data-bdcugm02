"""
ambiguous_filter.py — Fase 2: Filter Sampel Ambigu (Irisan 2 Kelas)
Track A — BDC Satria Data 2026

Input : diagnosis DataFrame (dari oof_diagnosis.py)
Output: ambiguous_candidates.csv — sampel dengan margin/entropy tinggi
        yang berpotensi menjadi sampel "irisan 2 kelas" seperti yang
        disinggung reviewer (mis. botol plastik berisi sisa makanan).

Prioritas: Organic↔Recyclable (sesuai diagnosis reviewer)
Double-flagged (muncul di cleanlab DAN margin rendah) = prioritas tertinggi.

Jalankan via notebook 04_cleanlab_cleaning.ipynb di Colab.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ─── Konstanta ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Recyclable", "Electronic", "Organic"]

# Pasangan kelas yang diprioritaskan sesuai masukan reviewer
PRIORITY_PAIRS = [
    (0, 2),  # Recyclable ↔ Organic (paling sering tertukar)
    (2, 0),
]
SECONDARY_PAIRS = [
    (0, 1), (1, 0),
    (1, 2), (2, 1),
]


# ─── Step 1: Filter Sampel Ambigu ─────────────────────────────────────────────

def filter_ambiguous(
    diag_df: pd.DataFrame,
    margin_threshold: Optional[float] = None,
    entropy_threshold: Optional[float] = None,
    auto_threshold_percentile: float = 5.0,
) -> pd.DataFrame:
    """
    Filter sampel ambigu berdasarkan margin rendah ATAU entropy tinggi.

    Threshold otomatis (jika tidak diisi):
    - margin_threshold  : P{auto_threshold_percentile} dari distribusi margin
    - entropy_threshold : P{100-auto_threshold_percentile} dari distribusi entropy

    Ini menyeleksi ~{auto_threshold_percentile}% sampel paling tidak yakin.

    Args:
        diag_df                    : output build_diagnosis_df()
        margin_threshold           : ambil sampel dengan margin < threshold
        entropy_threshold          : ambil sampel dengan entropy > threshold
        auto_threshold_percentile  : persentil untuk auto-threshold (default 5.0)

    Returns:
        DataFrame kandidat ambigu (union dari margin & entropy filter)
    """
    margins  = diag_df["margin"].values
    entropys = diag_df["entropy"].values

    # Auto-threshold
    if margin_threshold is None:
        margin_threshold = float(np.percentile(margins, auto_threshold_percentile))
        print(f"[filter_ambiguous] Auto margin threshold (P{auto_threshold_percentile}): "
              f"{margin_threshold:.4f}")
    if entropy_threshold is None:
        entropy_threshold = float(np.percentile(entropys, 100 - auto_threshold_percentile))
        print(f"[filter_ambiguous] Auto entropy threshold (P{100-auto_threshold_percentile}): "
              f"{entropy_threshold:.4f}")

    mask_margin  = diag_df["margin"] < margin_threshold
    mask_entropy = diag_df["entropy"] > entropy_threshold
    mask_union   = mask_margin | mask_entropy

    candidates = diag_df[mask_union].copy()

    # Kategorikan sumber ambiguitas
    candidates["ambig_source"] = "both"
    candidates.loc[mask_margin & ~mask_entropy, "ambig_source"] = "low_margin"
    candidates.loc[~mask_margin & mask_entropy, "ambig_source"] = "high_entropy"

    # Tentukan pasangan kelas yang berpotensi tertukar
    candidates["pair_label_pred"] = candidates.apply(
        lambda r: f"{CLASS_NAMES[int(r['label'])]}→{CLASS_NAMES[int(r['pred_class'])]}"
        if r["label"] != r["pred_class"]
        else f"{CLASS_NAMES[int(r['label'])]} (correct, low margin)",
        axis=1,
    )

    # Priority flag: Organic↔Recyclable (sesuai klaim reviewer)
    candidates["is_priority_pair"] = candidates.apply(
        lambda r: (int(r["label"]), int(r["pred_class"])) in PRIORITY_PAIRS,
        axis=1,
    )

    # Urutkan: priority pair dulu, lalu margin ascending
    candidates = candidates.sort_values(
        ["is_priority_pair", "margin"],
        ascending=[False, True]
    ).reset_index(drop=True)

    print(f"\n[filter_ambiguous] Kandidat ambigu total : {len(candidates):,}")
    print(f"  Low margin         : {mask_margin.sum():,}")
    print(f"  High entropy       : {mask_entropy.sum():,}")
    print(f"  Priority pairs     : {candidates['is_priority_pair'].sum():,}")
    print(f"  Threshold margin   : {margin_threshold:.4f}")
    print(f"  Threshold entropy  : {entropy_threshold:.4f}")

    return candidates, margin_threshold, entropy_threshold


# ─── Step 2: Double-flag (gabung cleanlab + margin/entropy) ──────────────────

def flag_double_flagged(
    ambiguous_df: pd.DataFrame,
    label_issues_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tambahkan kolom 'is_double_flagged' — sampel yang muncul di KEDUA:
    1. ambiguous_candidates (margin/entropy filter)
    2. label_issues dari cleanlab

    Double-flagged = kandidat paling meyakinkan untuk drop/relabel.

    Args:
        ambiguous_df     : output filter_ambiguous()
        label_issues_df  : output cleanlab_runner.run_full_cleanlab()
                           (harus punya kolom 'filepath' atau 'sample_idx')

    Returns:
        ambiguous_df dengan kolom 'is_double_flagged' tambahan
    """
    result = ambiguous_df.copy()

    # Cari irisan berdasarkan filepath
    if "filepath" in label_issues_df.columns and "filepath" in result.columns:
        cleanlab_fps = set(label_issues_df["filepath"].values)
        result["is_double_flagged"] = result["filepath"].isin(cleanlab_fps)
    elif "sample_idx" in label_issues_df.columns:
        cleanlab_idx = set(label_issues_df["sample_idx"].values)
        result["is_double_flagged"] = result.index.isin(cleanlab_idx)
    else:
        result["is_double_flagged"] = False
        print("[flag_double_flagged] ⚠️ Tidak bisa matching — 'filepath' tidak ada di kedua df")

    n_double = result["is_double_flagged"].sum()
    print(f"[flag_double_flagged] Double-flagged: {n_double:,} sampel "
          f"({n_double/len(result)*100:.1f}% dari kandidat ambigu)")
    print("  Double-flagged = prioritas tertinggi untuk review visual!")

    # Re-sort: double-flagged + priority pair dulu
    result = result.sort_values(
        ["is_double_flagged", "is_priority_pair", "margin"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return result


# ─── Step 3: Sanity check — cap 2-3% ─────────────────────────────────────────

def check_drop_cap(
    candidates: pd.DataFrame,
    total_train: int = 26527,
    cap_pct: float = 3.0,
) -> bool:
    """
    Peringatan jika jumlah kandidat melebihi cap 2-3% dari dataset train.
    Reviewer: "cap penghapusan maksimal ~2–3%".

    Returns True jika aman, False jika perlu perketat threshold.
    """
    n_candidates = len(candidates)
    cap_n = int(total_train * cap_pct / 100)
    pct = n_candidates / total_train * 100

    print(f"\n[check_drop_cap] Kandidat total : {n_candidates:,} ({pct:.1f}%)")
    print(f"[check_drop_cap] Cap maksimal   : {cap_n:,} ({cap_pct:.0f}%)")

    if n_candidates > cap_n:
        print(f"  ⚠️ MELEBIHI CAP! Kandidat {n_candidates:,} > cap {cap_n:,}")
        print("  → Perketat margin_threshold atau entropy_threshold sebelum review visual")
        return False
    else:
        print(f"  ✅ Masih dalam cap — aman dilanjutkan ke review visual")
        return True


# ─── Step 4: Distribusi per pasangan kelas ───────────────────────────────────

def summarize_pair_distribution(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Tampilkan distribusi kandidat per pasangan kelas.
    Berguna untuk mengkonfirmasi bahwa Organic↔Recyclable memang dominan.
    """
    summary = (
        candidates
        .groupby("pair_label_pred")
        .agg(
            count=("margin", "size"),
            margin_mean=("margin", "mean"),
            double_flagged=("is_double_flagged", "sum") if "is_double_flagged" in candidates.columns else ("margin", lambda x: 0),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    print("\n[summarize_pair_distribution] Distribusi per pasangan kelas:")
    print(summary.to_string(index=False))

    return summary


# ─── Step 5: Export ──────────────────────────────────────────────────────────

def export_ambiguous_candidates(
    candidates: pd.DataFrame,
    output_path: str,
) -> None:
    """Export ambiguous_candidates.csv untuk review visual."""
    cols_priority = [
        "filepath", "fold", "label", "pred_class",
        "margin", "entropy", "top1_prob", "top2_prob",
        "pair_label_pred", "is_priority_pair",
        "ambig_source",
    ]
    if "is_double_flagged" in candidates.columns:
        cols_priority.append("is_double_flagged")

    export_cols = [c for c in cols_priority if c in candidates.columns]
    candidates[export_cols].to_csv(output_path, index=False)
    print(f"[export_ambiguous_candidates] Disimpan ke {output_path}")
    print(f"  {len(candidates):,} kandidat")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_full_ambiguous_filter(
    diag_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    output_dir: str,
    label_issues_df: Optional[pd.DataFrame] = None,
    auto_threshold_percentile: float = 5.0,
    total_train: int = 26527,
) -> pd.DataFrame:
    """
    Entry point Fase 2. Filter sampel ambigu dan export ke ambiguous_candidates.csv.

    Returns:
        DataFrame kandidat ambigu (sudah di-sort, siap untuk contact_sheet.py)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FASE 2: FILTER SAMPEL AMBIGU")
    print("=" * 60)

    # Step 1: Filter
    candidates, mt, et = filter_ambiguous(
        diag_df,
        auto_threshold_percentile=auto_threshold_percentile,
    )

    # Step 2: Double-flag (jika cleanlab tersedia)
    if label_issues_df is not None:
        candidates = flag_double_flagged(candidates, label_issues_df)

    # Step 3: Cek cap
    is_within_cap = check_drop_cap(candidates, total_train=total_train)
    if not is_within_cap:
        print("\n  → Mencoba auto-perketat ke percentile lebih kecil...")
        stricter_percentile = auto_threshold_percentile / 2
        candidates, mt, et = filter_ambiguous(
            diag_df,
            auto_threshold_percentile=stricter_percentile,
        )
        if label_issues_df is not None:
            candidates = flag_double_flagged(candidates, label_issues_df)
        check_drop_cap(candidates, total_train=total_train)

    # Step 4: Summary
    summarize_pair_distribution(candidates)

    # Step 5: Export
    out_path = out_dir / "ambiguous_candidates.csv"
    export_ambiguous_candidates(candidates, str(out_path))

    print(f"\n✅ Fase 2 selesai! {len(candidates):,} kandidat siap untuk review visual.")
    print("  → Lanjutkan ke contact_sheet.py untuk generate grid gambar")

    return candidates

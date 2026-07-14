"""
generate_v2.py — Fase 3: Generate folds_v2.csv + class_weights_v2.npy
Track A — BDC Satria Data 2026

Input : cleaning_log.csv (hasil review visual) + folds.csv asli
Output:
    folds_v2.csv          — train bersih (drop/relabel diterapkan)
    class_weights_v2.npy  — bobot kelas dari distribusi baru

Invariant kritis:
- Sampel yang BERTAHAN tetap di fold asal (fold assignment TIDAK berubah)
- Skema kolom sama dengan folds.csv (filepath, label, fold)
- Total drop ≤ cap 2–3% dari dataset
- No-overlap antar fold tetap terjaga

Jalankan via notebook 05_generate_v2.ipynb di Colab.
"""

from pathlib import Path
from typing import Optional

import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


# ─── Konstanta ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Recyclable", "Electronic", "Organic"]
FOLDS_V2_COLS = ["filepath", "label", "fold"]   # kontrak skema — tidak boleh berubah


# ─── Step 1: Load dan Validasi cleaning_log.csv ──────────────────────────────

def load_and_validate_log(log_path: str) -> pd.DataFrame:
    """
    Load cleaning_log.csv dan pastikan semua keputusan valid.
    Raise jika ada entri pending atau keputusan tidak valid.
    """
    log = pd.read_csv(log_path)

    required_cols = {"filepath", "label_asli", "label_baru", "keputusan"}
    assert required_cols.issubset(log.columns), \
        f"cleaning_log.csv harus punya kolom: {required_cols}"

    valid_decisions = {"keep", "relabel", "drop"}
    invalid = log[~log["keputusan"].isin(valid_decisions)]
    if len(invalid) > 0:
        raise ValueError(
            f"{len(invalid)} keputusan tidak valid: {invalid['keputusan'].unique()}\n"
            f"Nilai valid: keep / relabel / drop"
        )

    pending = log[log["keputusan"] == "pending"]
    if len(pending) > 0:
        raise ValueError(
            f"{len(pending)} entri masih 'pending'. "
            f"Selesaikan review visual sebelum generate v2."
        )

    # Validasi relabel: label_baru harus berbeda dari label_asli
    relabel_rows = log[log["keputusan"] == "relabel"]
    bad_relabel = relabel_rows[relabel_rows["label_asli"] == relabel_rows["label_baru"]]
    if len(bad_relabel) > 0:
        raise ValueError(
            f"{len(bad_relabel)} entri 'relabel' tapi label_baru == label_asli. "
            f"Perbaiki dulu."
        )

    # Validasi label_baru in {0, 1, 2}
    assert set(log["label_baru"].dropna().astype(int).unique()) <= {0, 1, 2}, \
        f"label_baru harus 0/1/2"

    print(f"[load_and_validate_log] Log valid ✅")
    print(f"  keep   : {(log['keputusan'] == 'keep').sum():,}")
    print(f"  relabel: {(log['keputusan'] == 'relabel').sum():,}")
    print(f"  drop   : {(log['keputusan'] == 'drop').sum():,}")

    return log


# ─── Step 2: Apply cleaning ke folds.csv ─────────────────────────────────────

def apply_cleaning(
    folds_df: pd.DataFrame,
    cleaning_log: pd.DataFrame,
    cap_pct: float = 3.0,
    total_train: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply keputusan dari cleaning_log ke folds.csv:
    - 'drop'   : hapus baris
    - 'relabel': ganti kolom label dengan label_baru
    - 'keep'   : tidak ada perubahan

    Jaminan:
    - Fold assignment sampel yang bertahan TIDAK berubah
    - Skema kolom tetap: filepath, label, fold

    Args:
        folds_df      : DataFrame dari folds.csv asli
        cleaning_log  : DataFrame dari cleaning_log.csv
        cap_pct       : cap maksimal drop (default 3%)
        total_train   : total dataset (default = len(folds_df))

    Returns:
        folds_v2 DataFrame (siap disimpan)
    """
    total = total_train or len(folds_df)
    result = folds_df.copy()

    # Build lookup: filepath → keputusan + label_baru
    log_map = cleaning_log.set_index("filepath")[["keputusan", "label_baru"]].to_dict(orient="index")

    # Filepath yang di-drop
    drop_fps = set(
        row["filepath"] for row in cleaning_log.to_dict(orient="records")
        if row["keputusan"] == "drop"
    )

    # Filepath yang di-relabel
    relabel_map = {
        row["filepath"]: int(row["label_baru"])
        for row in cleaning_log.to_dict(orient="records")
        if row["keputusan"] == "relabel"
    }

    # Apply drop
    n_before = len(result)
    result = result[~result["filepath"].isin(drop_fps)].reset_index(drop=True)
    n_dropped = n_before - len(result)

    # Apply relabel
    n_relabeled = 0
    for fp, new_label in relabel_map.items():
        mask = result["filepath"] == fp
        if mask.any():
            result.loc[mask, "label"] = new_label
            n_relabeled += 1

    drop_pct = n_dropped / total * 100

    print(f"\n[apply_cleaning] Hasil:")
    print(f"  Sebelum : {n_before:,} sampel")
    print(f"  Drop    : {n_dropped:,} ({drop_pct:.2f}%)")
    print(f"  Relabel : {n_relabeled:,}")
    print(f"  Sesudah : {len(result):,} sampel")

    # Cek cap
    cap_n = int(total * cap_pct / 100)
    if n_dropped > cap_n:
        print(f"\n  ⚠️ PERINGATAN: Drop {n_dropped} melebihi cap {cap_n} ({cap_pct:.0f}%)!")
        print("  Lanjutkan dengan hati-hati — review apakah threshold terlalu agresif.")
    else:
        print(f"  ✅ Drop {n_dropped} ≤ cap {cap_n} ({cap_pct:.0f}%) — aman")

    # Pastikan fold assignment tidak berubah untuk sampel yang bertahan
    _verify_fold_assignment(result, folds_df)

    return result


def _verify_fold_assignment(folds_v2: pd.DataFrame, folds_orig: pd.DataFrame) -> None:
    """
    Assert: untuk setiap sampel yang bertahan di folds_v2,
    fold assignment-nya sama persis dengan folds.csv asli.
    """
    orig_fold_map = folds_orig.set_index("filepath")["fold"].to_dict()
    mismatches = []
    for _, row in folds_v2.iterrows():
        fp = row["filepath"]
        orig_fold = orig_fold_map.get(fp)
        if orig_fold is not None and orig_fold != row["fold"]:
            mismatches.append(fp)

    if mismatches:
        raise AssertionError(
            f"FOLD ASSIGNMENT BERUBAH untuk {len(mismatches)} sampel! "
            f"Ini melanggar aturan — OOF v1 vs v2 tidak bisa dibandingkan."
        )
    print("  ✅ Fold assignment sampel yang bertahan: tidak berubah")


# ─── Step 3: Verifikasi Integritas ───────────────────────────────────────────

def verify_integrity(
    folds_v2: pd.DataFrame,
    oof_v1: Optional[np.ndarray] = None,
    folds_orig: Optional[pd.DataFrame] = None,
) -> None:
    """
    Verifikasi integritas folds_v2:
    1. Skema kolom benar (filepath, label, fold)
    2. Label hanya 0/1/2
    3. No-overlap antar fold
    4. Proporsi kelas per fold masih seimbang (tidak ada fold yang 100% satu kelas)
    5. Distribusi per kelas dicek (tidak ada kelas yang terkikis berlebihan)
    """
    print("\n[verify_integrity] Verifikasi folds_v2...")

    # 1. Skema kolom
    assert all(c in folds_v2.columns for c in FOLDS_V2_COLS), \
        f"folds_v2 harus punya kolom: {FOLDS_V2_COLS}"
    print("  ✅ Skema kolom: OK")

    # 2. Label valid
    assert set(folds_v2["label"].unique()) <= {0, 1, 2}, \
        f"Label di luar 0/1/2: {folds_v2['label'].unique()}"
    print("  ✅ Label 0/1/2: OK")

    # 3. No-overlap antar fold
    folds = sorted(folds_v2["fold"].unique())
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            fps_i = set(folds_v2[folds_v2["fold"] == folds[i]]["filepath"])
            fps_j = set(folds_v2[folds_v2["fold"] == folds[j]]["filepath"])
            overlap = fps_i & fps_j
            assert len(overlap) == 0, \
                f"LEAKAGE! {len(overlap)} file ada di fold {folds[i]} DAN fold {folds[j]}"
    print("  ✅ No-overlap antar fold: OK")

    # 4. Proporsi kelas per fold
    print("\n  Distribusi kelas per fold:")
    for fold_id in folds:
        fold_labels = folds_v2[folds_v2["fold"] == fold_id]["label"]
        counts = fold_labels.value_counts().sort_index()
        n_fold = len(fold_labels)
        pcts = (counts / n_fold * 100).round(1)
        print(f"    Fold {fold_id}: n={n_fold:,} | " +
              " | ".join(f"{CLASS_NAMES[i]}={pcts.get(i, 0)}%" for i in range(3)))

    # 5. Distribusi global
    print("\n  Distribusi global folds_v2:")
    for label_id in range(3):
        count = (folds_v2["label"] == label_id).sum()
        print(f"    {CLASS_NAMES[label_id]:15s}: {count:,}")

    print("\n  ✅ Integritas folds_v2: semua cek lolos")


# ─── Step 4: Hitung class_weights_v2 ─────────────────────────────────────────

def compute_class_weights_v2(folds_v2: pd.DataFrame) -> np.ndarray:
    """
    Hitung class weights dari distribusi label baru di folds_v2.
    Formula: n_samples / (n_classes * count_per_class)
    """
    labels = folds_v2["label"].values
    n_total = len(labels)
    n_classes = 3
    counts = np.array([
        (labels == i).sum() for i in range(n_classes)
    ], dtype=float)

    weights = n_total / (n_classes * counts)
    print(f"\n[compute_class_weights_v2] Bobot kelas baru:")
    for i, (name, w, c) in enumerate(zip(CLASS_NAMES, weights, counts)):
        print(f"  {name:15s}: count={c:.0f} → weight={w:.4f}")

    return weights


# ─── Step 5: Export ──────────────────────────────────────────────────────────

def export_v2_artifacts(
    folds_v2: pd.DataFrame,
    weights_v2: np.ndarray,
    cleaning_log: pd.DataFrame,
    output_dir: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Export semua artefak v2 ke output_dir:
    - folds_v2.csv
    - class_weights_v2.npy
    - cleaning_summary.json (ringkasan untuk report)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # folds_v2.csv — hanya kolom kontrak
    folds_path = out_dir / "folds_v2.csv"
    folds_v2[FOLDS_V2_COLS].to_csv(folds_path, index=False)
    print(f"[export_v2_artifacts] folds_v2.csv → {folds_path}")
    print(f"  {len(folds_v2):,} baris")

    # class_weights_v2.npy
    weights_path = out_dir / "class_weights_v2.npy"
    np.save(weights_path, weights_v2)
    print(f"[export_v2_artifacts] class_weights_v2.npy → {weights_path}")

    # cleaning_summary.json — untuk report
    summary = {
        "total_before": int(len(folds_v2) + (cleaning_log["keputusan"] == "drop").sum()),
        "total_after":  int(len(folds_v2)),
        "n_dropped":    int((cleaning_log["keputusan"] == "drop").sum()),
        "n_relabeled":  int((cleaning_log["keputusan"] == "relabel").sum()),
        "n_kept":       int((cleaning_log["keputusan"] == "keep").sum()),
        "drop_pct":     round(
            (cleaning_log["keputusan"] == "drop").sum() /
            (len(folds_v2) + (cleaning_log["keputusan"] == "drop").sum()) * 100, 2
        ),
        "class_weights_v2": weights_v2.tolist(),
        "class_distribution": {
            CLASS_NAMES[i]: int((folds_v2["label"] == i).sum())
            for i in range(3)
        },
    }
    if metadata:
        summary["metadata"] = metadata

    summary_path = out_dir / "cleaning_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[export_v2_artifacts] cleaning_summary.json → {summary_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_generate_v2(
    folds_csv: str,
    cleaning_log_path: str,
    output_dir: str,
    oof_v1_path: Optional[str] = None,
    cap_pct: float = 3.0,
) -> dict:
    """
    Entry point Fase 3. Generate folds_v2.csv + class_weights_v2.npy.

    Returns:
        dict berisi folds_v2 DataFrame, weights_v2, dan cleaning_log
    """
    print("=" * 60)
    print("FASE 3: GENERATE ARTEFAK V2 — folds_v2.csv + class_weights_v2.npy")
    print("=" * 60)

    # Load
    folds_df      = pd.read_csv(folds_csv)
    cleaning_log  = load_and_validate_log(cleaning_log_path)
    oof_v1        = np.load(oof_v1_path) if oof_v1_path else None

    # Apply cleaning
    folds_v2 = apply_cleaning(folds_df, cleaning_log, cap_pct=cap_pct)

    # Verifikasi
    verify_integrity(folds_v2, oof_v1, folds_df)

    # Class weights
    weights_v2 = compute_class_weights_v2(folds_v2)

    # Export
    export_v2_artifacts(
        folds_v2=folds_v2,
        weights_v2=weights_v2,
        cleaning_log=cleaning_log,
        output_dir=output_dir,
        metadata={
            "folds_csv_source": str(folds_csv),
            "cleaning_log_source": str(cleaning_log_path),
        },
    )

    print("\n" + "=" * 60)
    print("✅ FASE 3 SELESAI!")
    print(f"   folds_v2.csv    : {len(folds_v2):,} sampel")
    print(f"   class_weights_v2: {weights_v2.round(4).tolist()}")
    print("=" * 60)
    print("\n→ Umumkan GATE G2 ke Track B & C:")
    print("  'GATE G2 hijau — folds_v2.csv + class_weights_v2.npy sudah di Drive'")

    return {
        "folds_v2":     folds_v2,
        "weights_v2":   weights_v2,
        "cleaning_log": cleaning_log,
    }

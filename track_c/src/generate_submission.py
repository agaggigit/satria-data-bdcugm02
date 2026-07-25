"""generate_submission.py — Pipeline end-to-end Track C (v2, embedding-based).

Urutan langkah:
  0. Assert label mapping (safety check)
  1. Validate OOF (shape, alignment, no NaN)
  2. Nested validation → estimasi JUJUR gain threshold (wajib sebelum submit)
  3. Threshold tuning di seluruh OOF (untuk threshold final yang dipakai ke test)
  4. Ensemble inference dari embedding cache → prob_test [1458, 3]
  5. Apply threshold → label [1458]
  6. Generate submission_apace.csv
  7. Validator format (wajib lolos)
  8. Buat dan simpan manifest
"""
import os
import sys
import numpy as np
import pandas as pd

# Pastikan repo root di sys.path
_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from track_c.src.config_c import CFG_C
from track_c.src.inference import assert_label_mapping
from track_c.src.threshold_tuning import tune_thresholds_oof, apply_thresholds
from track_c.src.nested_validation import nested_cv_threshold
from track_c.src.ensemble_embedding import run_ensemble_inference
from track_c.src.validator import validate_submission
from track_c.src.manifest import build_manifest, save_manifest, print_manifest_summary


def validate_oof_file(oof_path: str, folds_df: pd.DataFrame) -> np.ndarray:
    """Load dan validasi OOF: shape, alignment, no NaN, prob sum ~1."""
    if not os.path.exists(oof_path):
        raise FileNotFoundError(
            f"OOF file tidak ditemukan: {oof_path}\n"
            f"Pastikan Track B sudah menyerahkan file ini ke Drive."
        )

    oof = np.load(oof_path)
    n_train = len(folds_df)

    # Shape
    assert oof.shape == (n_train, 3), (
        f"Shape OOF salah: {oof.shape} (harusnya ({n_train}, 3))\n"
        f"folds_csv punya {n_train} baris. Pastikan OOF align dengan folds_v2.csv!"
    )

    # NaN / inf
    assert not np.isnan(oof).any(), "OOF mengandung NaN!"
    assert np.isfinite(oof).all(), "OOF mengandung inf!"

    # Prob sum ≈ 1
    sums = oof.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-3), (
        f"OOF prob tidak sum ke 1 (max_abs_err={np.max(np.abs(sums - 1)):.6f})"
    )

    print(f"  ✅ OOF valid: shape={oof.shape}, "
          f"baseline argmax F1 dihitung di step 2")
    return oof


def main(
    submission_number: int = 2,
    ensemble_weights: dict = None,
    notes: str = "",
):
    """Jalankan pipeline Track C lengkap.

    Args:
        submission_number : 1, 2, atau 3
        ensemble_weights  : dict {comp_key: weight} atau None (seragam)
        notes             : catatan untuk manifest
    """
    print("=" * 60)
    print(f"PIPELINE TRACK C — Submission {submission_number}")
    print(f"Tim: {CFG_C.team_name}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 0. Safety check label mapping
    # -----------------------------------------------------------------------
    print("\n[STEP 0] ASSERT LABEL MAPPING")
    assert_label_mapping(CFG_C)
    print("  ✅ Label mapping: 0=Recyclable, 1=Electronic, 2=Organic")

    # -----------------------------------------------------------------------
    # 1. Load OOF + folds
    # -----------------------------------------------------------------------
    print("\n[STEP 1] LOAD OOF & FOLDS")
    if not os.path.exists(CFG_C.folds_csv):
        raise FileNotFoundError(
            f"folds_v2.csv tidak ditemukan: {CFG_C.folds_csv}\n"
            "Pastikan Track A sudah menyerahkan folds_v2.csv ke Drive."
        )
    folds_df = pd.read_csv(CFG_C.folds_csv)
    print(f"  folds_v2.csv: {len(folds_df)} baris, "
          f"label dist: {dict(folds_df['label'].value_counts().sort_index())}")

    # Load OOF dari komposisi aktif pertama (untuk threshold tuning)
    # Kalau multi-komposisi, OOF gabungan dirata-rata dulu
    active = CFG_C.active_compositions
    if len(active) == 1:
        oof_path = CFG_C.compositions[active[0]].oof
        oof = validate_oof_file(oof_path, folds_df)
        print(f"  OOF: {os.path.basename(oof_path)}")
    else:
        # Multi-komposisi: rata-rata OOF dengan bobot ensemble_weights
        oof_list = []
        for key in active:
            comp_oof_path = CFG_C.compositions[key].oof
            o = validate_oof_file(comp_oof_path, folds_df)
            oof_list.append(o)
        if ensemble_weights is None:
            w_arr = [1.0 / len(oof_list)] * len(oof_list)
        else:
            total = sum(ensemble_weights.get(k, 1) for k in active)
            w_arr = [ensemble_weights.get(k, 1) / total for k in active]
        oof = sum(w * o for w, o in zip(w_arr, oof_list))
        print(f"  OOF: rata-rata {len(active)} komposisi dengan bobot {w_arr}")

    true_labels = folds_df["label"].values
    fold_assignments = folds_df["fold"].values

    # -----------------------------------------------------------------------
    # 2. Nested validation (wajib — estimasi jujur sebelum commit ke threshold)
    # -----------------------------------------------------------------------
    print("\n[STEP 2] NESTED VALIDATION")
    nested_result = nested_cv_threshold(
        oof_probs=oof,
        true_labels=true_labels,
        fold_assignments=fold_assignments,
        lo=CFG_C.threshold_search_lo,
        hi=CFG_C.threshold_search_hi,
        n_steps=CFG_C.nested_cv_steps,
        verbose=True,
    )

    # Peringatan kalau gap terlalu besar
    if nested_result["illusion_gap"] > 0.003:
        print("\n  ⚠️  PERINGATAN: illusion_gap > 0.003 — threshold mungkin overfit ke OOF!")
        print("      Pertimbangkan pakai argmax (threshold=[1.0, 1.0, 1.0]) sebagai gantinya.")

    # -----------------------------------------------------------------------
    # 3. Threshold tuning di seluruh OOF (untuk dipakai ke test)
    # -----------------------------------------------------------------------
    print("\n[STEP 3] THRESHOLD TUNING")
    best_thresholds = tune_thresholds_oof(
        oof_probs=oof,
        true_labels=true_labels,
        n_steps=CFG_C.threshold_search_steps,
    )

    # -----------------------------------------------------------------------
    # 4. Ensemble inference dari embedding cache
    # -----------------------------------------------------------------------
    print("\n[STEP 4] ENSEMBLE INFERENCE (embedding-based)")
    test_probs = run_ensemble_inference(
        cfg=CFG_C,
        folds_df=folds_df,
        ensemble_weights=ensemble_weights,
    )

    if test_probs is None or len(test_probs) == 0:
        raise RuntimeError("Inference gagal: tidak ada prediksi yang dihasilkan.")

    # -----------------------------------------------------------------------
    # 5. Apply threshold
    # -----------------------------------------------------------------------
    print("\n[STEP 5] APPLY THRESHOLDS")
    print(f"  Threshold: {best_thresholds}")
    final_labels = apply_thresholds(test_probs, best_thresholds)
    label_counts = dict(zip(*np.unique(final_labels, return_counts=True)))
    print(f"  Distribusi prediksi: {label_counts}")

    # Sanity: distribusi prediksi harus masuk akal
    for cls in [0, 1, 2]:
        frac = label_counts.get(int(cls), 0) / len(final_labels)
        if frac < 0.01:
            print(f"  ⚠️  Kelas {cls} cuma {frac:.1%} dari prediksi — mencurigakan!")

    # -----------------------------------------------------------------------
    # 6. Generate submission CSV
    # -----------------------------------------------------------------------
    print("\n[STEP 6] GENERATE SUBMISSION CSV")
    template_df = pd.read_csv(CFG_C.sample_sub_path)
    assert len(template_df) == len(test_probs), (
        f"Jumlah baris template ({len(template_df)}) "
        f"!= jumlah prediksi ({len(test_probs)}). "
        "Cek embedding test alignment!"
    )

    sub_df = template_df.copy()
    sub_df["predicted"] = final_labels.astype(int)

    # Nama file: submission_apace.csv
    team_suffix = f"_{CFG_C.team_name}" if CFG_C.team_name else ""
    submission_filename = f"submission{team_suffix}.csv"
    os.makedirs(CFG_C.output_dir, exist_ok=True)
    submission_path = os.path.join(CFG_C.output_dir, submission_filename)
    sub_df.to_csv(submission_path, index=False)
    print(f"  File: {submission_path}")

    # -----------------------------------------------------------------------
    # 7. Validator format
    # -----------------------------------------------------------------------
    print("\n[STEP 7] FORMAT VALIDATION")
    is_valid = validate_submission(submission_path, CFG_C.sample_sub_path)
    if not is_valid:
        raise RuntimeError(
            "SUBMISSION TIDAK VALID! Perbaiki bug sebelum upload.\n"
            "File TIDAK akan diunggah — ini safeguard untuk mencegah jatah hangus."
        )

    # -----------------------------------------------------------------------
    # 8. Manifest
    # -----------------------------------------------------------------------
    print("\n[STEP 8] BUILD MANIFEST")
    manifest = build_manifest(
        submission_number=submission_number,
        submission_path=submission_path,
        cfg=CFG_C,
        active_compositions=active,
        thresholds=best_thresholds,
        nested_result=nested_result,
        ensemble_weights=ensemble_weights,
        notes=notes,
    )
    save_manifest(manifest, CFG_C.output_dir, submission_number)
    print_manifest_summary(manifest)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"✅ SELESAI — Submission {submission_number} siap diunggah!")
    print(f"   File    : {submission_path}")
    print(f"   Estimasi: nested CV mean = {nested_result['nested_mean']:.5f}")
    print(f"   Threshold: {best_thresholds}")
    print("=" * 60)
    print("\n⚠️  INGAT:")
    print("   1. Unggah ≤ 28 Juli (panen) atau ≤ 29 Juli (asuransi)")
    print("   2. Tie-break: yang unggah LEBIH DULU menang kalau skor sama")
    print("   3. Validator sudah lolos — JANGAN unggah file yang berbeda")

    return submission_path, manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Track C — Generate Submission")
    parser.add_argument("--submission-number", type=int, default=2)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    main(submission_number=args.submission_number, notes=args.notes)

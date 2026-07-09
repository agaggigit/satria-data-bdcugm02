import os
import torch
import numpy as np
import pandas as pd
from track_c.src.config_c import CFG_C
from track_c.src.inference import assert_label_mapping
from track_c.src.threshold_tuning import tune_thresholds_oof, apply_thresholds
from track_c.src.ensemble_tta import run_5fold_ensemble_inference
from track_c.src.validator import validate_submission

def main():
    print("=" * 60)
    print(f"PIPELINE EVALUASI & SUBMISSION — Tim {CFG_C.team_name}")
    print("=" * 60)

    # Auto-detect device: pakai GPU jika tersedia, fallback ke CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device.upper()}")

    # 0. Safety Check
    assert_label_mapping(CFG_C)

    # 1. Tuning Threshold di OOF
    print("\n[STEP 1] THRESHOLD TUNING")
    oof_path = CFG_C.oof_npy
    folds_df = pd.read_csv(CFG_C.folds_csv)

    if not os.path.exists(oof_path):
        print(f"ERROR: OOF file tidak ditemukan di {oof_path}")
        return

    oof_probs = np.load(oof_path)
    true_labels = folds_df['label'].values

    best_thresholds = tune_thresholds_oof(oof_probs, true_labels)

    # 2. Ensemble & TTA di Test Data
    print("\n[STEP 2] ENSEMBLE & TTA INFERENCE PADA TEST DATA")

    test_probs = run_5fold_ensemble_inference(device=device)

    if test_probs is None or len(test_probs) == 0:
        print("ERROR: Inference gagal, tidak ada prediksi yang dihasilkan.")
        return

    # 3. Apply Threshold ke Test Probs
    print("\n[STEP 3] APPLY THRESHOLDS")
    final_labels = apply_thresholds(test_probs, best_thresholds)

    # 4. Generate Submission CSV
    print("\n[STEP 4] GENERATE SUBMISSION CSV")
    sub_df = pd.read_csv(CFG_C.sample_sub_path)
    sub_df['predicted'] = final_labels

    # Nama file: 'submission_NamaTim.csv' sesuai konvensi plan & checklist format
    team_suffix = f"_{CFG_C.team_name}" if CFG_C.team_name else ""
    submission_filename = f"submission{team_suffix}.csv"

    # Pastikan output_dir ada
    os.makedirs(CFG_C.output_dir, exist_ok=True)
    submission_path = os.path.join(CFG_C.output_dir, submission_filename)
    sub_df.to_csv(submission_path, index=False)
    print(f"Submission file dibuat: {submission_path}")

    # 5. Validasi Format
    print("\n[STEP 5] FORMAT VALIDATION")
    is_valid = validate_submission(submission_path, CFG_C.sample_sub_path)

    if is_valid:
        print(f"\nSELAMAT! File submission siap diunggah: {submission_path}")
    else:
        print("\nPERINGATAN: File submission tidak valid. Tolong perbaiki sebelum unggah.")

if __name__ == "__main__":
    main()

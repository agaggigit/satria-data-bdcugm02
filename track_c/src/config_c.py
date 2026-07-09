import os
from types import SimpleNamespace

# Base path folder utama di Google Drive
DRIVE_BASE_PATH = "/content/drive/MyDrive/BDC2026 apace"

# Sub-folder per track (sesuai struktur Drive yang terlihat di foto)
OUTPUT_TRACK_A = os.path.join(DRIVE_BASE_PATH, "output_trackA")
OUTPUT_TRACK_B = os.path.join(DRIVE_BASE_PATH, "output_trackB")
OUTPUT_TRACK_C = os.path.join(DRIVE_BASE_PATH, "output_trackC")

CFG_C = SimpleNamespace(
    # Reproducibility (harus konsisten dengan Track A & B)
    seed=42,

    # Dataset params
    img_size=224,
    num_classes=3,
    class_names=["Recyclable", "Electronic", "Organic"],
    label_map={0: "Recyclable", 1: "Electronic", 2: "Organic"},

    # Model params (konsisten dengan Track B)
    backbone="convnext_tiny.in12k_ft_in1k",
    drop_path_rate=0.1,

    # Paths ke artefak Track A
    folds_csv=os.path.join(OUTPUT_TRACK_A, "folds.csv"),
    test_dir=os.path.join(OUTPUT_TRACK_A, "test"),

    # Template submission dari panitia - ada di root BDC2026 apace/
    sample_sub_path=os.path.join(DRIVE_BASE_PATH, "submission.csv"),

    # Paths ke output Track B
    oof_npy=os.path.join(OUTPUT_TRACK_B, "oof.npy"),
    checkpoints_dir=OUTPUT_TRACK_B,

    # Paths output untuk Track C
    output_dir=OUTPUT_TRACK_C,
    team_name="apace",  # Nama tim -> file output: submission_apace.csv

    # Inference config
    use_tta=True,
    batch_size=32,
    num_workers=2
)

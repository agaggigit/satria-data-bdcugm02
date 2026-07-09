import os
from types import SimpleNamespace

# Mengambil base path dari Google Drive, karena user bilang semua diletakkan di 1 folder: 'BDC2026 apace'
# Pada Google Colab, path MyDrive adalah /content/drive/MyDrive/
DRIVE_BASE_PATH = "/content/drive/MyDrive/BDC2026 apace"

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
    drop_path_rate=0.1,  # Untuk test kita juga butuh define arsitektur yang sama persis
    
    # Paths ke artefak
    # (Karena semua ditaruh di 1 folder, kita point ke folder yang sama)
    folds_csv=os.path.join(DRIVE_BASE_PATH, "folds.csv"),
    test_dir=os.path.join(DRIVE_BASE_PATH, "test"),
    sample_sub_path=os.path.join(DRIVE_BASE_PATH, "submission.csv"),
    
    # Paths ke output Track B
    oof_npy=os.path.join(DRIVE_BASE_PATH, "oof.npy"),
    checkpoints_dir=DRIVE_BASE_PATH,
    
    # Paths output untuk Track C
    output_dir=DRIVE_BASE_PATH,
    team_name="",  # Dikosongkan → file output: submission.csv
    
    # Inference config
    use_tta=True,
    batch_size=32,
    num_workers=2
)

import os
from types import SimpleNamespace

# Base path folder utama di Google Drive (sesuai struktur Drive tim: BDC2026apace/)
DRIVE_BASE_PATH = "/content/drive/MyDrive/BDC2026apace"
OUTPUT_TRACK_A = os.path.join(DRIVE_BASE_PATH, "output_trackA")
OUTPUT_TRACK_B = os.path.join(DRIVE_BASE_PATH, "output_trackB")

CFG = SimpleNamespace(
    # === DARI FASE 0 (tidak dihapus) ===
    # Reproducibility
    seed=42,
    # Data
    img_size=224,
    batch=64,               # naik dari 32 — grad checkpointing dimatikan, T4 15GB masih longgar (lihat model.py)
    accum_steps=1,          # >1 kalau naik img_size (mis. 288) & batch fisik harus turun di T4
    num_workers=2,          # Colab: 2 CPU cores
    grad_checkpointing=False,  # OFF: ConvNeXt-Tiny @224 cuma pakai ~2GB/15GB, checkpointing cuma bikin lambat tanpa manfaat VRAM
    num_classes=3,
    class_names=["Recyclable", "Electronic", "Organic"],  # index = label
    label_map={0: "Recyclable", 1: "Electronic", 2: "Organic"},
    # Model
    backbone="convnext_tiny.in12k_ft_in1k",
    drop_path_rate=0.1,
    # Optimizer
    lr=3e-4,
    min_lr=1e-6,
    weight_decay=0.05,
    layer_decay=None,       # None = LLRD mati. Isi mis. 0.9 utk eksperimen Task 7 (lihat optim_utils.py)
    # Schedule
    epochs=8,
    warmup_epochs=1,
    patience=None,          # None = tidak ada early stopping. Isi mis. 4 utk eksperimen epoch>=20
    # Loss
    label_smoothing=0.1,
    # Stability
    max_grad_norm=1.0,

    # === BARU FASE 1 (ditambah) ===
    # Paths — sesuai struktur Drive tim asli (BDC2026apace/output_trackA|B)
    folds_csv=os.path.join(OUTPUT_TRACK_A, "folds.csv"),
    class_weights_path=os.path.join(OUTPUT_TRACK_A, "class_weights.npy"),
    save_dir=OUTPUT_TRACK_B,
)

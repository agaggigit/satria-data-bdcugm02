"""config_c.py — Konfigurasi Track C (v2, embedding-based pipeline).

Perubahan v2 vs v1:
- Tidak lagi asumsi satu backbone ConvNeXt + checkpoint .pt
- Mendukung multi-backbone via COMPOSITIONS (dict OOF + embedding test per backbone/head)
- OOF sekarang diakses via COMPOSITIONS, bukan oof_npy tunggal
- Inference test: load emb_*_test.npy → re-fit head → predict_proba (bukan forward model besar)
"""
import os
from types import SimpleNamespace

# =========================================================================
# Base paths (sesuaikan kalau lokasi Drive berbeda)
# =========================================================================
DRIVE_BASE_PATH = "/content/drive/MyDrive/BDC2026apace"
OUTPUT_TRACK_A  = os.path.join(DRIVE_BASE_PATH, "output_trackA")
OUTPUT_TRACK_B  = os.path.join(DRIVE_BASE_PATH, "output_trackB")
OUTPUT_TRACK_C  = os.path.join(DRIVE_BASE_PATH, "output_trackC")
EMB_DIR         = os.path.join(OUTPUT_TRACK_B, "embeddings")

# =========================================================================
# Komposisi tersedia — SATU ENTRY PER OOF yang sudah ada di Drive.
# Kunci   : label bebas, tapi deskriptif (dipakai di manifest & log)
# oof     : path ke file oof_*.npy (prob [N_train, 3], align folds_v2.csv)
# emb_test: path ke file embedding test [1458, D]
# head    : nama head yang dipakai ('knn', 'linear', 'mlp', 'lgbm')
# backbone: nama model asli (untuk dokumentasi report & manifest)
# =========================================================================
COMPOSITIONS = {
    "siglip2_knn": SimpleNamespace(
        oof      = os.path.join(OUTPUT_TRACK_B, "oof_siglip2so400m_knn_v2.npy"),
        emb_test = os.path.join(EMB_DIR, "siglip2so400m_test.npy"),
        head     = "knn",
        backbone = "google/siglip2-so400m-patch14-384",
        img_size = 384,   # SigLIP2-SO400M native size
    ),
    # Tambahkan komposisi lain saat OOF-nya datang dari Track B, contoh:
    # "dinov3_linear": SimpleNamespace(
    #     oof      = os.path.join(OUTPUT_TRACK_B, "oof_dinov3_linear_v2.npy"),
    #     emb_test = os.path.join(EMB_DIR, "dinov3_test.npy"),
    #     head     = "linear",
    #     backbone = "facebook/dinov2-large",
    #     img_size = 518,
    # ),
}

# Komposisi aktif untuk submission saat ini — ubah sesuai keputusan tim
ACTIVE_COMPOSITIONS = ["siglip2_knn"]

# =========================================================================
# Konfigurasi utama Track C
# =========================================================================
CFG_C = SimpleNamespace(
    # Reproducibility
    seed=42,

    # Label mapping — WAJIB konsisten dengan Track A & B
    num_classes=3,
    class_names=["Recyclable", "Electronic", "Organic"],
    label_map={0: "Recyclable", 1: "Electronic", 2: "Organic"},

    # Paths artefak Track A
    folds_csv=os.path.join(OUTPUT_TRACK_A, "folds_v2.csv"),        # data bersih, 25985 baris
    folds_csv_original=os.path.join(OUTPUT_TRACK_A, "folds.csv"),  # asli 26527 baris (align embedding)

    # Template submission panitia
    sample_sub_path=os.path.join(DRIVE_BASE_PATH, "submission.csv"),

    # Output Track C
    output_dir=OUTPUT_TRACK_C,
    team_name="apace",   # → file output: submission_apace.csv

    # Threshold tuning — langkah 0.05 sesuai plan v3
    threshold_search_steps=30,          # linspace(0.5, 2.0, 30) ≈ langkah 0.052
    threshold_search_lo=0.5,
    threshold_search_hi=2.0,

    # Nested validation
    nested_cv_steps=30,                 # sama dengan threshold_search_steps

    # Compositions (referensi ke dict di atas)
    compositions=COMPOSITIONS,
    active_compositions=ACTIVE_COMPOSITIONS,

    # Nama file OOF aktif (kompatibilitas backward dengan kode lama)
    # Diisi otomatis dari COMPOSITIONS[ACTIVE_COMPOSITIONS[0]] kalau ada 1 komposisi
    oof_npy=COMPOSITIONS[ACTIVE_COMPOSITIONS[0]].oof if ACTIVE_COMPOSITIONS else None,
)

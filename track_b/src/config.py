import os
from dataclasses import dataclass, field, replace

# Base path folder utama di Google Drive (sesuai struktur Drive tim: BDC2026apace/)
DRIVE_BASE_PATH = "/content/drive/MyDrive/BDC2026apace"
OUTPUT_TRACK_A = os.path.join(DRIVE_BASE_PATH, "output_trackA")
OUTPUT_TRACK_B = os.path.join(DRIVE_BASE_PATH, "output_trackB")


@dataclass(frozen=True)
class Config:
    # === DARI FASE 0 (tidak dihapus) ===
    seed: int = 42
    img_size: int = 224
    batch: int = 64             # naik dari 32 — grad checkpointing dimatikan, T4 15GB masih longgar
    accum_steps: int = 1        # >1 kalau naik img_size (mis. 288) & batch fisik harus turun di T4
    num_workers: int = 2        # Colab: 2 CPU cores
    grad_checkpointing: bool = False  # OFF: ConvNeXt-Tiny @224 cuma ~2GB/15GB, checkpointing cuma bikin lambat
    num_classes: int = 3
    class_names: list = field(default_factory=lambda: ["Recyclable", "Electronic", "Organic"])
    label_map: dict = field(default_factory=lambda: {0: "Recyclable", 1: "Electronic", 2: "Organic"})

    # Model
    backbone: str = "convnext_tiny.in12k_ft_in1k"
    drop_path_rate: float = 0.1
    pretrained: bool = True     # False di test/audit wiring supaya tidak unduh bobot

    # Optimizer
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    layer_decay: float = None   # None = LLRD mati. Isi mis. 0.9 (lihat optim_utils.py)

    # Schedule
    epochs: int = 8
    warmup_epochs: int = 1
    patience: int = None        # None = tidak ada early stopping. Isi mis. 4 utk eksperimen epoch>=20

    # Loss
    label_smoothing: float = 0.1
    # Stability
    max_grad_norm: float = 1.0

    # === BARU FASE 1 ===
    # Paths — sesuai struktur Drive tim asli (BDC2026apace/output_trackA|B)
    # FOLDS_CSV_OVERRIDE: dibaca kalau ada (mis. folds.csv hasil copy gambar ke
    # storage lokal Colab, filepath sudah diarahkan lokal -- baca gambar dari
    # Drive per-file jauh lebih lambat daripada dari disk lokal). Opt-in, tidak
    # mengubah perilaku default kalau env var tidak di-set.
    folds_csv: str = os.environ.get("FOLDS_CSV_OVERRIDE", os.path.join(OUTPUT_TRACK_A, "folds.csv"))
    class_weights_path: str = os.path.join(OUTPUT_TRACK_A, "class_weights.npy")
    save_dir: str = OUTPUT_TRACK_B

    # === BARU — versioning checkpoint (fix Fase 2/3) ===
    run_name: str = "convnext_v1"   # WAJIB diganti tiap run baru (retrain v2, eksperimen HP, family)
    allow_overwrite: bool = False   # guard; True hanya kalau sengaja menimpa checkpoint sendiri
    vflip: bool = False             # dipakai src/transforms.py (Task 4), belum aktif di Task 1

    # === BARU — embedding-first pipeline (Track_B embedding implementation plan) ===
    # Drive, bukan storage lokal runtime -- kalau sesi Colab putus, cache embedding
    # (berjam-jam kerja GPU) tidak boleh ikut hilang.
    embeddings_dir: str = os.path.join(OUTPUT_TRACK_B, "embeddings")

    # Nama field disamakan dengan track_c/src/config_c.py supaya konsisten lintas
    # track meski config-nya terpisah per track (bukan shared).
    test_dir: str = os.path.join(DRIVE_BASE_PATH, "test")
    sample_sub_path: str = os.path.join(DRIVE_BASE_PATH, "submission.csv")


CFG = Config()


def make_cfg(**overrides) -> Config:
    """Salinan CFG dengan sebagian field diganti. Tidak memutasi CFG dasar.

    frozen=True bukan formalitas: setattr(cfg, "img_sizee", 288) diam-diam bikin
    atribut baru dan run tetap jalan di 224 tanpa error. replace(cfg, img_sizee=288)
    langsung TypeError. Salah ketik nama field harus gagal keras, bukan menghasilkan
    eksperimen yang keliru tapi kelihatan valid.
    """
    return replace(CFG, **overrides)

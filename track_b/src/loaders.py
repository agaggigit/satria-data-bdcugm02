import pandas as pd
from torch.utils.data import DataLoader

from dataset import WasteDataset          # Track A -- read-only, lihat track_a/src/dataset.py
from transforms import build_transforms


def get_loaders_b(fold: int, cfg, data_config: dict, val_batch_size: int | None = None):
    """Loader Track B: reuse Dataset Track A, transform disuntik per-backbone.

    Return (train_loader, val_loader, val_row_idx).
    val_row_idx = index baris folds.csv untuk fold ini, URUT sesuai urutan val_loader
    (val_loader shuffle=False) -- ini yang dipakai assemble_oof() untuk align baris.
    """
    import os
    df = pd.read_csv(cfg.folds_csv)

    # Otomatis alihkan ke storage lokal SSD Colab jika /tmp/dataset/train tersedia
    local_dir = "/tmp/dataset/train"
    if os.path.exists(local_dir):
        def _redirect(p):
            if "/content/drive/" in p and "train/" in p:
                sub_path = p.split("train/")[-1]
                return os.path.join(local_dir, sub_path)
            return p
        df["filepath"] = df["filepath"].apply(_redirect)
        print(f"⚡ [Fast I/O] Path gambar dialihkan ke lokal SSD Colab: {local_dir}")
    tr_df = df[df["fold"] != fold]
    va_df = df[df["fold"] == fold]

    train_tfm = build_transforms(data_config, cfg.img_size, train=True, vflip=cfg.vflip)
    eval_tfm = build_transforms(data_config, cfg.img_size, train=False)

    train_ds = WasteDataset(tr_df, transform=train_tfm)
    val_ds = WasteDataset(va_df, transform=eval_tfm)

    val_bs = val_batch_size or getattr(cfg, "val_batch", max(cfg.batch, 32))
    num_workers = getattr(cfg, "num_workers", 0)
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_bs, shuffle=False,   # WAJIB False -> alignment OOF
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=persistent_workers,
    )

    val_row_idx = va_df.index.to_numpy()
    assert len(val_row_idx) == len(val_ds), "val_row_idx tidak sepanjang val dataset"
    return train_loader, val_loader, val_row_idx

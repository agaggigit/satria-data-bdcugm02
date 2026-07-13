import pandas as pd
from torch.utils.data import DataLoader

from dataset import WasteDataset          # Track A -- read-only, lihat track_a/src/dataset.py
from transforms import build_transforms


def get_loaders_b(fold: int, cfg, data_config: dict):
    """Loader Track B: reuse Dataset Track A, transform disuntik per-backbone.

    Return (train_loader, val_loader, val_row_idx).
    val_row_idx = index baris folds.csv untuk fold ini, URUT sesuai urutan val_loader
    (val_loader shuffle=False) -- ini yang dipakai assemble_oof() untuk align baris.
    """
    df = pd.read_csv(cfg.folds_csv)
    tr_df = df[df["fold"] != fold]
    va_df = df[df["fold"] == fold]

    train_tfm = build_transforms(data_config, cfg.img_size, train=True, vflip=cfg.vflip)
    eval_tfm = build_transforms(data_config, cfg.img_size, train=False)

    train_ds = WasteDataset(tr_df, transform=train_tfm)
    val_ds = WasteDataset(va_df, transform=eval_tfm)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch, shuffle=False,   # WAJIB False -> alignment OOF
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )

    val_row_idx = va_df.index.to_numpy()
    assert len(val_row_idx) == len(val_ds), "val_row_idx tidak sepanjang val dataset"
    return train_loader, val_loader, val_row_idx

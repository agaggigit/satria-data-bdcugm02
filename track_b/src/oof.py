"""oof.py — Rakit & validasi OOF dengan alignment baris yang ketat.

Beda dengan oof_utils.py (isi incremental per fold via fill_fold_oof): modul
ini menerima seluruh fold_probs sekaligus dan mengecek setiap baris hanya
diisi tepat satu kali oleh fold yang tidak melatihnya — mengunci bug paling
senyap di pipeline OOF (baris tertukar / index tidak align dengan folds.csv).
"""
import numpy as np
import pandas as pd
import torch


def assemble_oof(fold_probs: dict, n_rows: int) -> np.ndarray:
    """fold_probs[fold] = (row_indices ke folds.csv, probs [Nf, 3])."""
    oof = np.full((n_rows, 3), np.nan, dtype=np.float64)
    for fold, (idx, probs) in fold_probs.items():
        idx = np.asarray(idx)
        assert len(idx) == len(probs), f"fold {fold}: jumlah index != jumlah probs"
        assert np.isnan(oof[idx]).all(), f"fold {fold}: menimpa baris yang sudah terisi"
        oof[idx] = probs
    return oof


def validate_oof(oof: np.ndarray, folds_df: pd.DataFrame) -> None:
    assert oof.shape == (len(folds_df), 3), f"shape OOF {oof.shape} != ({len(folds_df)}, 3)"
    n_missing = int(np.isnan(oof).any(axis=1).sum())
    assert n_missing == 0, f"{n_missing} baris OOF belum terisi"
    sums = oof.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-3), "probabilitas OOF tidak sum ke 1"
    assert set(folds_df["label"].unique()) <= {0, 1, 2}, "label di luar mapping 0/1/2"


@torch.no_grad()
def predict_fold(model, loader, device) -> tuple:
    """Return (row_indices, probs). Loader HARUS mengembalikan (images, labels, row_idx)."""
    model.eval()
    all_idx, all_probs = [], []
    for images, _labels, row_idx in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_idx.append(np.asarray(row_idx))
    return np.concatenate(all_idx), np.concatenate(all_probs)

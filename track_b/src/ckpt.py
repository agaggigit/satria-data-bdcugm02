"""ckpt.py — Path checkpoint/history berbasis run_name + guard anti-overwrite.

Sebelum ini, train.py menyimpan ke nama tetap (fold{N}.pt) -- retrain apa pun
(v2, eksperimen HP) diam-diam menimpa checkpoint sebelumnya, termasuk v1 yang
masih dipakai Track C. Fungsi di sini memaksa nama file membawa run_name, dan
menolak menimpa file yang sudah ada kecuali diminta eksplisit.
"""
from pathlib import Path

import torch

REQUIRED_CKPT_KEYS = {
    "model_state_dict",
    "model_name",
    "data_config",   # tanpa ini, Track C harus menebak preprocessing -> sumber bug senyap
    "run_name",
    "fold",
    "img_size",
    "seed",
}


def checkpoint_path(cfg, fold: int) -> Path:
    return Path(cfg.save_dir) / f"{cfg.run_name}_fold{fold}.pt"


def history_path(cfg, fold: int) -> Path:
    return Path(cfg.save_dir) / f"{cfg.run_name}_fold{fold}_history.json"


def save_checkpoint(payload: dict, path, allow_overwrite: bool = False) -> None:
    missing = REQUIRED_CKPT_KEYS - set(payload)
    if missing:
        raise ValueError(f"payload checkpoint kehilangan key wajib: {sorted(missing)}")

    path = Path(path)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"{path} sudah ada. Ganti cfg.run_name, atau set cfg.allow_overwrite=True "
            f"kalau kamu memang sengaja menimpanya."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

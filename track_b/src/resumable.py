"""resumable.py — Cache generik skip-if-exists untuk pasangan (array.npy + meta.json).

Dipakai head_grid_v3.py dan fair_compare_v1_v2.py (TRACK_B_ARAHAN_V3.md Task 2-3)
supaya run yang ke-interupsi (Colab free tier memutus sesi idle) tidak
menghitung ulang dari nol -- konsisten dengan filosofi is_cached/save_shard di
embed.py, tapi generik untuk kunci apa pun (bukan cuma nama embedding).
"""
import json
import os

import numpy as np


def run_or_load(out_dir: str, stem: str, compute_fn) -> tuple:
    """compute_fn() -> (array, meta_dict). Kalau `{stem}.npy` + `{stem}_meta.json`
    dua-duanya sudah ada di out_dir, load balik -- compute_fn TIDAK dipanggil
    sama sekali. Cache dianggap sah hanya kalau keduanya ada (sama prinsipnya
    dengan embed.is_cached -- .npy tanpa meta bisa berarti proses mati di
    tengah jalan, jangan dikira sudah lengkap)."""
    npy_path = os.path.join(out_dir, f"{stem}.npy")
    meta_path = os.path.join(out_dir, f"{stem}_meta.json")

    if os.path.exists(npy_path) and os.path.exists(meta_path):
        arr = np.load(npy_path)
        with open(meta_path) as f:
            meta = json.load(f)
        return arr, meta

    arr, meta = compute_fn()
    os.makedirs(out_dir, exist_ok=True)
    np.save(npy_path, arr)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return arr, meta

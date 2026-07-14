"""extract_all.py — Ekstrak semua backbone kandidat (GPU).

Jalankan dengan `python experiments/extract_all.py` dari mana saja -- sys.path
di-set eksplisit di bawah (jangan andalkan cwd: `python path/to/script.py`
menaruh folder SCRIPT itu sendiri di sys.path[0], bukan cwd tempat kamu `cd`).

DINOv3 gated -- terima lisensinya di HF dan siapkan token SEBELUM menjalankan
ini, bukan setelah 40 menit ekstraksi gagal di tengah.
"""
import os
import sys

import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

from config import CFG
from embed import assert_aligned, extract_embeddings, save_embeddings

BACKBONES = {
    "siglip2b256":  "google/siglip2-base-patch16-256",
    "siglip2so400m": "google/siglip2-so400m-patch14-384",
    "siglip1b256":  "google/siglip-base-patch16-256",
    "dinov3vitl":   "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3cnxb":   "facebook/dinov3-convnext-base-pretrain-lvd1689m",
}

folds = pd.read_csv(CFG.folds_csv)
template = pd.read_csv(CFG.sample_sub_path)
test_paths = [f"{CFG.test_dir}/{i}.jpg" for i in template["id"]]

for name, ckpt in BACKBONES.items():
    for flips, suffix in [((), ""), (("h", "v"), "_tta")]:
        for split, paths in [("train", folds["filepath"].tolist()), ("test", test_paths)]:
            emb = extract_embeddings(ckpt, paths, batch=32, flips=flips)
            if split == "train":
                assert_aligned(emb, folds)
            save_embeddings(emb, name + suffix, split,
                            {"checkpoint": ckpt, "flips": list(flips), "seed": CFG.seed})
            print(f"{name}{suffix} {split}: {emb.shape}")

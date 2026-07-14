"""probe_grid.py — Grid backbone x head x TTA -> tabel keputusan (CPU, menit).

Jalankan dengan `python experiments/probe_grid.py` dari mana saja -- sys.path
di-set eksplisit di bawah (jangan andalkan cwd: `python path/to/script.py`
menaruh folder SCRIPT itu sendiri di sys.path[0], bukan cwd tempat kamu `cd`).
RESULTS_PATH tetap relatif ke CWD (bukan ke lokasi script) -- jalankan dari
track_b/src/ seperti skrip lain supaya "../results/probe_grid.csv" tepat sasaran.

Cara memilih pemenang -- JANGAN asal ambil `mean` tertinggi:
1. Urutkan berdasarkan `mean`, tapi buang kandidat yang `std` antar-fold-nya
   jauh lebih besar dari yang lain (gain-nya tidak stabil).
2. Di antara kandidat yang `mean`-nya beda tipis (< 0.002), pilih yang `min`
   (fold terburuk) paling tinggi -- itu yang paling mungkin bertahan di test.
3. Kalau `mean` naik tapi `min` turun -> TOLAK. Itu tanda mengejar satu fold
   yang beruntung, bukan gain yang sungguhan.
"""
import os
import sys

import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

from config import CFG
from consistency import fold_consistency
from embed import load_embeddings
from features import concat_features, l2norm
from heads import HEAD_NAMES
from probe_cv import run_probe_cv

RESULTS_PATH = "../results/probe_grid.csv"

folds = pd.read_csv(CFG.folds_csv)
SINGLES = ["siglip2b256", "siglip2so400m", "siglip1b256", "dinov3vitl", "dinov3cnxb"]

rows = []

# --- backbone tunggal x head ---
for name in SINGLES:
    X = l2norm(load_embeddings(name, "train")[0])
    for head in HEAD_NAMES:
        oof, _ = run_probe_cv(X, folds, head, class_weight="balanced", seed=CFG.seed)
        c = fold_consistency(oof, folds)
        rows.append({"combo": name, "head": head, **c})
        np.save(f"oof_{name}_{head}.npy", oof)
        print(f"{name:16s} {head:7s} mean={c['mean']:.4f} min={c['min']:.4f} std={c['std']:.4f}")

# --- concat: batasi kombinasinya (anti overfit-OOF) ---
CONCATS = [
    ("siglip2so400m", "dinov3vitl"),                  # image-text x self-supervised
    ("siglip2so400m", "dinov3vitl", "dinov3cnxb"),    # + CNN bias
    ("siglip2so400m", "siglip1b256", "dinov3vitl"),
]
for combo in CONCATS:
    X = concat_features([load_embeddings(n, "train")[0] for n in combo])
    for head in ["linear", "mlp"]:
        oof, _ = run_probe_cv(X, folds, head, class_weight="balanced", seed=CFG.seed)
        c = fold_consistency(oof, folds)
        rows.append({"combo": "+".join(combo), "head": head, **c})
        print(f"{'+'.join(combo):40s} {head:7s} mean={c['mean']:.4f} min={c['min']:.4f}")

df = pd.DataFrame(rows).sort_values("mean", ascending=False)
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
df.to_csv(RESULTS_PATH, index=False)
print(df.head(10).to_string(index=False))

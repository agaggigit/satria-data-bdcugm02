"""safety_net.py — Jalur tercepat ke submission 1 (GPU).

Ekstrak embedding SigLIP2 sekali, linear-probe 5-fold, lalu prediksi test set.
Jalankan dengan `python experiments/safety_net.py` dari mana saja -- sys.path
di-set eksplisit di bawah (jangan andalkan cwd: `python path/to/script.py`
menaruh folder SCRIPT itu sendiri di sys.path[0], bukan cwd tempat kamu `cd`).

Urutan penting:
1. Mirror gambar ke disk lokal, ekstrak embedding train, assert alignment, cache.
2. Linear probe 5-fold -> OOF -- INI angka penentu, kirim ke Track A SEKARANG.
3. Test set HANYA dipakai untuk prediksi akhir (tidak untuk fit/tuning apa pun).

Semua artefak ditulis ke Drive (CFG.save_dir), bukan cwd runtime Colab yang
hilang begitu sesi putus.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

from config import CFG
from embed import assert_aligned, extract_embeddings, is_cached, load_embeddings, save_embeddings
from features import l2norm
from heads import make_head
from local_cache import localize_paths
from metrics import macro_f1
from probe_cv import run_probe_cv
from submission import make_submission, validate_submission

CKPT = "google/siglip2-base-patch16-256"
NAME = "siglip2b256"
LOCAL_ROOT = "/content/img_cache"

folds = pd.read_csv(CFG.folds_csv)
template = pd.read_csv(CFG.sample_sub_path)
y = folds["label"].to_numpy()

# --- 1. Embedding train (urutan HARUS folds.csv) ---
if is_cached(NAME, "train"):
    emb_tr, _ = load_embeddings(NAME, "train")
    print(f"train: pakai cache {CFG.embeddings_dir}")
else:
    train_paths = localize_paths(folds["filepath"].tolist(),
                                 os.path.join(LOCAL_ROOT, "train"), desc="copy train")
    emb_tr = extract_embeddings(CKPT, train_paths)
    save_embeddings(emb_tr, NAME, "train",
                    {"checkpoint": CKPT, "flips": [], "seed": CFG.seed})
assert_aligned(emb_tr, folds)
print("dim embedding:", emb_tr.shape)      # cetak; jangan percaya asumsi

# --- 2. Linear probe 5-fold: ANGKA PENENTU ---
X = l2norm(emb_tr)
oof, scores = run_probe_cv(X, folds, "linear", class_weight="balanced", seed=CFG.seed)

for i, s in enumerate(scores):
    print(f"fold {i}: {s:.4f}")
cv_mean, cv_std = float(np.mean(scores)), float(np.std(scores))
oof_overall = macro_f1(y, oof.argmax(1))
print(f"CV = {cv_mean:.4f} +/- {cv_std:.4f}")
print(f"OOF overall = {oof_overall:.4f}")

oof_path = os.path.join(CFG.save_dir, "oof_probe.npy")
np.save(oof_path, oof)
print(f"oof_probe.npy tersimpan: {oof_path}")   # <-- KIRIM KE TRACK A SEKARANG JUGA

# Skor ditulis ke file, bukan cuma di-print: output cell Colab bisa hilang, dan
# angka ini tidak bisa direkonstruksi dari oof.npy tanpa menghitung ulang.
summary = {
    "backbone": CKPT, "short_name": NAME, "head": "linear",
    "class_weight": "balanced", "seed": CFG.seed,
    "folds": {f"fold{i}": float(s) for i, s in enumerate(scores)},
    "cv_mean_macro_f1": cv_mean, "cv_std_macro_f1": cv_std,
    "oof_overall_macro_f1_argmax": float(oof_overall),
    "n_rows": int(len(folds)), "dim": int(emb_tr.shape[1]),
    "folds_csv": CFG.folds_csv,
}
summary_path = os.path.join(CFG.save_dir, "cv_summary_safety_net.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"skor tersimpan: {summary_path}")

# --- 3. Test -> submission (test HANYA untuk prediksi akhir) ---
if is_cached(NAME, "test"):
    emb_te, _ = load_embeddings(NAME, "test")
    print(f"test: pakai cache {CFG.embeddings_dir}")
else:
    test_paths = localize_paths([f"{CFG.test_dir}/{i}.jpg" for i in template["id"]],
                                os.path.join(LOCAL_ROOT, "test"), desc="copy test")
    emb_te = extract_embeddings(CKPT, test_paths)
    save_embeddings(emb_te, NAME, "test",
                    {"checkpoint": CKPT, "flips": [], "seed": CFG.seed})

head = make_head("linear", seed=CFG.seed, class_weight="balanced").fit(X, y)   # fit di SELURUH train
pred = head.predict_proba(l2norm(emb_te)).argmax(axis=1)

sub = make_submission(pred, template)
validate_submission(sub, template)
sub_path = os.path.join(CFG.save_dir, "submission_apace.csv")
sub.to_csv(sub_path, index=False)
print(f"VALIDATOR LOLOS — siap diunggah: {sub_path}")

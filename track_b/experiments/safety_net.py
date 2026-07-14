"""safety_net.py — Jalur tercepat ke submission 1 (GPU, ~1 jam total).

Ekstrak embedding SigLIP2 sekali, linear-probe 5-fold, lalu prediksi test set.
Jalankan dari track_b/src/ (sys.path harus di sini supaya import flat jalan).

Urutan penting:
1. Ekstrak embedding train, assert alignment ke folds.csv, simpan cache.
2. Linear probe 5-fold -> OOF -- INI angka penentu, kirim ke Track A SEKARANG.
3. Test set HANYA dipakai untuk prediksi akhir (tidak untuk fit/tuning apa pun).
"""
import numpy as np
import pandas as pd

from config import CFG
from embed import assert_aligned, extract_embeddings, load_embeddings, save_embeddings
from features import l2norm
from heads import make_head
from metrics import macro_f1
from probe_cv import run_probe_cv
from submission import make_submission, validate_submission

CKPT = "google/siglip2-base-patch16-256"
NAME = "siglip2b256"

folds = pd.read_csv(CFG.folds_csv)
template = pd.read_csv(CFG.sample_sub_path)

# --- 1. Ekstrak train (urutan HARUS folds.csv) ---
emb_tr = extract_embeddings(CKPT, folds["filepath"].tolist())
assert_aligned(emb_tr, folds)
print("dim embedding:", emb_tr.shape)      # cetak; jangan percaya asumsi
save_embeddings(emb_tr, NAME, "train",
                {"checkpoint": CKPT, "flips": [], "seed": CFG.seed})

# --- 2. Linear probe 5-fold: ANGKA PENENTU ---
X = l2norm(emb_tr)
y = folds["label"].to_numpy()
oof, scores = run_probe_cv(X, folds, "linear", class_weight="balanced", seed=CFG.seed)

for i, s in enumerate(scores):
    print(f"fold {i}: {s:.4f}")
print(f"CV = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
print(f"OOF overall = {macro_f1(y, oof.argmax(1)):.4f}")

np.save("oof_probe.npy", oof)   # <-- KIRIM KE TRACK A SEKARANG JUGA

# --- 3. Test -> submission (test HANYA untuk prediksi akhir) ---
test_paths = [f"{CFG.test_dir}/{i}.jpg" for i in template["id"]]   # SESUAIKAN pola nama file
emb_te = extract_embeddings(CKPT, test_paths)
save_embeddings(emb_te, NAME, "test", {"checkpoint": CKPT, "flips": [], "seed": CFG.seed})

head = make_head("linear", seed=CFG.seed, class_weight="balanced").fit(X, y)   # fit di SELURUH train
pred = head.predict_proba(l2norm(emb_te)).argmax(axis=1)

sub = make_submission(pred, template)
validate_submission(sub, template)
sub.to_csv("submission_apace.csv", index=False)
print("VALIDATOR LOLOS — siap diunggah.")

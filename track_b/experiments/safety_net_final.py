"""safety_net_final.py — Hasilkan submission FINAL dari komposisi pemenang.

SUBMISSION MASIH 0/3 -- ini jalur tercepat ke file yang lolos validator, siap
diunggah MANUAL ke portal panitia (script TIDAK mengunggah apa pun).

Komposisi (divalidasi 5-fold CV lewat probe_grid.py + uji TTA terpisah):
    backbone = siglip2so400m   (embedding beku, sudah di-cache di Drive)
    head     = kNN (k=15, cosine, weights=distance)  -- dari heads.py
    TTA      = TIDAK (delta mean dalam noise, min turun -> ditolak)
    CV       = mean 0.9901 | min 0.9896 | std 0.0005

Beda dari CV: di sini kNN di-fit di SELURUH train (bukan per-fold) karena ini
untuk memprediksi test final, bukan mengukur generalisasi.

Semua logika dipakai ulang dari src/ (load_embeddings, l2norm, make_head,
make_submission, validate_submission) -- tidak ada yang ditulis ulang.
Jalankan: `python experiments/safety_net_final.py` (CPU, detik).
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

# Ringan (numpy/pandas saja) -> aman di-import test tanpa menyeret torch/sklearn.
from submission import make_submission, validate_submission

BACKBONE = "siglip2so400m"
HEAD = "knn"
USE_TTA = False

# Nama tim: sumber kebenaran track_c/src/config_c.py (team_name="apace"), konsisten
# dengan submission_apace.csv yang sudah ada. Bukan tebakan.
TEAM_NAME = "apace"

# Angka CV dari probe_grid.py (5-fold OOF) -- untuk manifest/jejak audit, TIDAK
# dihitung ulang di sini (script ini fit-full-train untuk prediksi test).
CV_MEAN, CV_MIN, CV_STD = 0.9901, 0.9896, 0.0005


def build_validated_submission(pred: np.ndarray, template_df: pd.DataFrame) -> pd.DataFrame:
    """make_submission + gerbang validasi eksplisit, lalu kembalikan df siap tulis.

    validate_submission (src/submission.py) SUDAH menegakkan keempat syarat yang
    diminta: 1458 baris, urutan id == template, tidak ada NaN, nilai in {0,1,2}.
    Dipakai apa adanya sebagai gerbang -- tidak ditulis ulang. Dipisah dari IO
    supaya bisa diuji CPU-only tanpa memuat embedding 26.527 baris."""
    sub = make_submission(pred, template_df)
    validate_submission(sub, template_df)   # raise AssertionError kalau salah satu syarat gagal
    return sub


def main() -> None:
    # Import berat ditaruh di sini, bukan top-level -> test cukup import
    # build_validated_submission tanpa torch/sklearn.
    from config import CFG
    from embed import load_embeddings
    from features import l2norm
    from heads import make_head

    folds = pd.read_csv(CFG.folds_csv)
    template = pd.read_csv(CFG.sample_sub_path)
    assert "id" in template.columns, \
        f"template {CFG.sample_sub_path} tidak punya kolom 'id': {list(template.columns)}"

    # --- 1-3. Train: load emb -> L2-norm -> fit kNN di SELURUH train ---
    emb_tr = load_embeddings(BACKBONE, "train")[0]
    assert emb_tr.shape[0] == len(folds), \
        f"emb train {emb_tr.shape[0]} != folds {len(folds)} -- alignment rusak"
    X_tr = l2norm(emb_tr)
    y = folds["label"].to_numpy()

    # KNeighborsClassifier TIDAK punya param class_weight; make_head('knn')
    # mengabaikannya kalau dikirim. Jadi TIDAK dikirim -- eksplisit, bukan silent.
    head = make_head(HEAD, seed=CFG.seed)
    head.fit(X_tr, y)

    # --- 4-5. Test: L2-norm CARA SAMA PERSIS (fungsi sama, tanpa param beda) ---
    emb_te = load_embeddings(BACKBONE, "test")[0]
    assert emb_te.shape[0] == len(template), \
        f"emb test {emb_te.shape[0]} != template {len(template)} -- alignment rusak"
    X_te = l2norm(emb_te)
    pred = head.predict_proba(X_te).argmax(axis=1).astype(int)

    # --- 6. Gerbang validasi eksplisit sebelum menulis apa pun ---
    sub = build_validated_submission(pred, template)

    # --- 7. Tulis submission_NamaTim.csv ---
    out_csv = os.path.join(CFG.save_dir, f"submission_{TEAM_NAME}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    sub.to_csv(out_csv, index=False)

    # --- 8. Manifest kecil di sebelahnya (asal-usul + jejak audit) ---
    manifest = {
        "backbone": BACKBONE,
        "head": HEAD,
        "tta": USE_TTA,
        "cv_mean": CV_MEAN,
        "cv_min": CV_MIN,
        "cv_std": CV_STD,
        "seed": CFG.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path = os.path.join(CFG.save_dir, f"submission_{TEAM_NAME}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    dist = pd.Series(pred).value_counts().sort_index().to_dict()
    print(f"backbone={BACKBONE} head={HEAD} tta={USE_TTA} "
          f"CV mean={CV_MEAN} min={CV_MIN} std={CV_STD}")
    print(f"distribusi prediksi test (0/1/2): {dist}")
    print(f"VALIDATOR LOLOS -- {len(sub)} baris, kolom {list(sub.columns)}")
    print(f"submission : {out_csv}")
    print(f"manifest   : {manifest_path}")
    print("\nSIAP DIUNGGAH MANUAL KE PORTAL PANITIA")


if __name__ == "__main__":
    main()

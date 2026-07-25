"""handoff_v3.py — Task 4 (TRACK_B_ARAHAN_V3.md): serahkan OOF kandidat
pemenang ke Track C sebagai artefak kanonis oof.npy + oof_meta.json.

Kontrak Track C (Workflow_Koordinasi_ABC.md): oof.npy = probabilitas [N,3],
index cocok folds.csv (v1, N=26.527). Kalau pemenang berasal dari
folds_version=2 (dilatih di data v2), OOF mentahnya TIDAK bisa langsung
diserahkan -- panjangnya cuma len(folds_v2), bukan len(folds.csv). Untuk kasus
itu dipakai varian 'adil' dari fair_compare_v1_v2.py (train v2, eval val-fold
v1 UTUH) yang SUDAH align ke folds.csv v1 -- itu yang benar diserahkan, bukan
OOF naif v2.
"""
import json
import os

import numpy as np
import pandas as pd

from metrics import macro_f1


def select_source_oof_path(cache_dir: str, combo: str, head: str, folds_version: int) -> tuple:
    """Path (oof.npy, meta.json) sumber untuk kandidat pemenang.
    version==1 -> langsung dari head_grid_v3.py.
    version==2 -> varian 'adil' dari fair_compare_v1_v2.py (satu-satunya OOF
    v2-trained yang align ke folds.csv v1; OOF naif v2 TIDAK align)."""
    if folds_version == 1:
        stem = f"oof_{combo}_{head}_v1"
    elif folds_version == 2:
        stem = f"faircmp_{combo}_{head}_adil_v2"
    else:
        raise ValueError(f"folds_version harus 1 atau 2, dapat: {folds_version}")
    return (os.path.join(cache_dir, f"{stem}.npy"),
            os.path.join(cache_dir, f"{stem}_meta.json"))


def handoff_winner(cache_dir: str, combo: str, head: str, folds_version: int,
                   folds_v1: pd.DataFrame, out_dir: str,
                   allow_overwrite: bool = False, extra_meta: dict = None) -> dict:
    """Salin OOF kandidat pemenang jadi oof.npy + oof_meta.json kanonis di
    out_dir. Guard anti-overwrite (pola sama dgn ckpt.save_checkpoint) dan
    guard alignment (panjang HARUS == len(folds_v1), tidak ada NaN, sum ke 1)
    -- kalau salah satu gagal, TIDAK ADA yang ditulis."""
    npy_src, meta_src = select_source_oof_path(cache_dir, combo, head, folds_version)
    if not (os.path.exists(npy_src) and os.path.exists(meta_src)):
        hint = " / fair_compare_v1_v2.py" if folds_version == 2 else ""
        raise FileNotFoundError(
            f"Sumber OOF pemenang tidak ada: {npy_src}. Jalankan head_grid_v3.py{hint} dulu."
        )

    oof = np.load(npy_src)
    n_expected = len(folds_v1)
    assert oof.shape == (n_expected, 3), (
        f"OOF pemenang {oof.shape} != ({n_expected}, 3) -- TIDAK align ke folds.csv v1, "
        f"TIDAK BOLEH diserahkan ke Track C apa adanya"
    )
    assert not np.isnan(oof).any(), "OOF pemenang mengandung NaN -- batal handoff"
    assert np.allclose(oof.sum(axis=1), 1.0, atol=1e-3), \
        "OOF pemenang tidak sum ke 1 -- batal handoff"

    out_npy = os.path.join(out_dir, "oof.npy")
    out_meta = os.path.join(out_dir, "oof_meta.json")
    if (os.path.exists(out_npy) or os.path.exists(out_meta)) and not allow_overwrite:
        raise FileExistsError(
            f"{out_npy} (atau oof_meta.json) sudah ada -- kemungkinan handoff sebelumnya "
            f"(mis. dari era ConvNeXt/kNN). Set allow_overwrite=True kalau memang sengaja menimpanya."
        )

    with open(meta_src) as f:
        src_meta = json.load(f)

    y_v1 = folds_v1["label"].to_numpy()
    overall_f1 = macro_f1(y_v1, oof.argmax(axis=1))

    meta = {
        "shape": list(oof.shape),
        "index_source": "folds.csv (v1) baris ke-i oof = baris ke-i folds.csv",
        "content": "softmax probabilities kelas 0/1/2",
        "combo": combo, "head": head, "folds_version_trained_on": folds_version,
        "oof_overall_macro_f1_argmax": float(overall_f1),
        "source_stem": os.path.basename(npy_src)[:-4],
        "source_cv_mean": src_meta.get("cv_mean"),
        "source_cv_min": src_meta.get("cv_min"),
        "source_cv_std": src_meta.get("cv_std"),
        "seed": src_meta.get("seed"),
    }
    if extra_meta:
        meta.update(extra_meta)

    os.makedirs(out_dir, exist_ok=True)
    np.save(out_npy, oof.astype(np.float32))
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    return meta

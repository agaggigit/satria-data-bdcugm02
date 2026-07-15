"""tta_compare.py — Apakah TTA (_tta, hflip+vflip dirata-rata di level embedding)
benar-benar menaikkan skor? Uji HANYA kandidat teratas, bukan seluruh grid.

Konteks: probe_grid.py sudah menguji 5 backbone x 4 head varian NON-TTA. Pemenang
saat ini: siglip2so400m + knn (mean=0.9901, min=0.9896). Yang belum terjawab:
apakah varian _tta layak dipakai. Menjalankan ulang seluruh grid untuk itu buang
waktu -- datanya sudah ada di probe_grid.csv. Script ini fokus:

  siglip2so400m x {knn, mlp}  ->  non-TTA (dari file oof lama) vs TTA (dihitung baru)

Non-TTA TIDAK dihitung ulang: skornya dibaca dari oof_{backbone}_{head}.npy yang
sudah ditulis probe_grid.py. Hanya TTA yang perlu CV baru.

Aturan keputusan = SAMA dengan probe_grid.py (anti overfit-OOF): gain kecil yang
konsisten di semua fold > gain besar yang cuma di satu fold. TTA cuma dipakai
kalau mean naik DAN fold terburuk (min) tidak turun.

Jalankan: `python experiments/tta_compare.py` (CPU, detik). Logika di src/ dipakai
ulang apa adanya -- tidak ada yang ditulis ulang.
"""
import os
import sys

import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

BACKBONE = "siglip2so400m"
HEADS = ["knn", "mlp"]                 # dua teratas dari grid non-TTA


def decide_tta(mean_notta: float, min_notta: float,
               mean_tta: float, min_tta: float) -> str:
    """Aturan identik probe_grid.py. TTA hanya menang kalau mean naik DAN fold
    terburuk tidak ikut turun -- kalau mean naik tapi min turun, itu mengejar
    satu fold beruntung (overfit ke OOF), bukan gain sungguhan."""
    delta = mean_tta - mean_notta
    if delta <= 0:
        return "TIDAK MEMBANTU, buang TTA"
    if min_tta >= min_notta:
        return "TTA dipakai"
    return "TOLAK (overfit ke OOF)"


def build_row(backbone: str, head: str, c_notta: dict, c_tta: dict) -> dict:
    """Rangkai satu baris perbandingan dari dua hasil fold_consistency().
    Pisah dari IO supaya bisa diuji CPU-only dengan data mock."""
    return {
        "backbone": backbone,
        "head": head,
        "mean_notta": c_notta["mean"],
        "mean_tta": c_tta["mean"],
        "min_notta": c_notta["min"],
        "min_tta": c_tta["min"],
        "delta": c_tta["mean"] - c_notta["mean"],
        "keputusan": decide_tta(c_notta["mean"], c_notta["min"],
                                c_tta["mean"], c_tta["min"]),
    }


def main() -> None:
    # Import berat ditaruh di sini (bukan top-level) supaya test bisa meng-import
    # decide_tta/build_row tanpa menyeret CFG, sklearn, lightgbm, dsb.
    from config import CFG
    from consistency import fold_consistency
    from embed import load_embeddings
    from features import l2norm
    from probe_cv import run_probe_cv

    folds = pd.read_csv(CFG.folds_csv)
    rows = []

    for head in HEADS:
        # --- NON-TTA: baca dari oof lama, JANGAN re-train ---
        notta_path = os.path.join(CFG.save_dir, f"oof_{BACKBONE}_{head}.npy")
        if not os.path.exists(notta_path):
            raise FileNotFoundError(
                f"OOF non-TTA tidak ada: {notta_path}\n"
                f"Jalankan probe_grid.py dulu (yang menulis oof_{BACKBONE}_{head}.npy), "
                f"atau cek nama file sebenarnya sebelum lanjut."
            )
        oof_notta = np.load(notta_path)
        c_notta = fold_consistency(oof_notta, folds)

        # --- TTA: hitung CV baru dari embedding _tta (class_weight sama spt probe_grid) ---
        emb_tta = load_embeddings(f"{BACKBONE}_tta", "train")[0]
        X_tta = l2norm(emb_tta)
        oof_tta, _ = run_probe_cv(X_tta, folds, head,
                                  class_weight="balanced", seed=CFG.seed)
        c_tta = fold_consistency(oof_tta, folds)

        # Simpan oof TTA (pola nama sama, cuma tambah _tta)
        tta_oof_path = os.path.join(CFG.save_dir, f"oof_{BACKBONE}_tta_{head}.npy")
        np.save(tta_oof_path, oof_tta)

        row = build_row(BACKBONE, head, c_notta, c_tta)
        rows.append(row)

        print(f"=== {BACKBONE} + {head} ===")
        print(f"  non-TTA : mean={c_notta['mean']:.4f} min={c_notta['min']:.4f} std={c_notta['std']:.4f}")
        print(f"  TTA     : mean={c_tta['mean']:.4f} min={c_tta['min']:.4f} std={c_tta['std']:.4f}")
        print(f"  delta mean (TTA - non-TTA) = {row['delta']:+.4f}")
        print(f"  KEPUTUSAN: {row['keputusan']}")
        print(f"  oof TTA -> {tta_oof_path}\n")

    df = pd.DataFrame(rows, columns=["backbone", "head", "mean_notta", "mean_tta",
                                     "min_notta", "min_tta", "delta", "keputusan"])

    # Repo (bisa di-commit) + Drive best-effort (repo di /content ephemeral).
    repo_csv = os.path.join(os.path.dirname(__file__), "..", "results", "tta_comparison.csv")
    os.makedirs(os.path.dirname(repo_csv), exist_ok=True)
    df.to_csv(repo_csv, index=False)
    print(f"perbandingan tersimpan: {repo_csv}")

    try:
        drive_csv = os.path.join(CFG.save_dir, "tta_comparison.csv")
        os.makedirs(os.path.dirname(drive_csv), exist_ok=True)
        df.to_csv(drive_csv, index=False)
    except OSError:
        pass

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

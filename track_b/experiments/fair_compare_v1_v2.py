"""fair_compare_v1_v2.py — Task 3 (TRACK_B_ARAHAN_V3.md B3/A4): cleaned (v2)
vs non-cleaned (v1), dibandingkan ADIL -- bukan cuma "v2 CV-nya lebih tinggi".

Kenapa perbandingan naif menyesatkan (B3): kalau v2 dievaluasi HANYA di baris
yang tersisa (val fold v2), CV-nya bisa naik semu -- sampel sulit sudah
dibuang, soalnya jadi lebih gampang. Itu bukan bukti cleaning membantu.

Tiga angka dilaporkan berdampingan per head, SEMUA di eval set yang identik
(val fold v1 utuh) kecuali yang ditandai "naif":
  1. baseline  : train v1, eval val-fold v1 utuh       -- acuan sebelum cleaning
  2. naif      : train v2, eval val-fold v2 (subset)   -- TIDAK sebanding v1
  3. adil      : train v2, eval val-fold v1 UTUH        -- sebanding apple-to-apple
                 (termasuk baris yang di-drop dari v2 -- model tidak pernah
                 dilatih dgn baris-baris fold itu, baik dari v1 maupun v2)

delta_naif_minus_adil = berapa banyak "kenaikan" v2 cuma karena eval jadi lebih
mudah. delta_adil_minus_baseline = gain SUNGGUHAN dari cleaning (data lebih
bersih), diukur di kesulitan eval yang sama persis dengan v1.

Drop rate Track A ~2.3% (di bawah cap 3%) -- risiko inflasi CV relatif rendah,
TAPI tetap dijalankan (jangan dilewati hanya karena angkanya terlihat aman).

Jalankan dari track_b/src/ (sama seperti skrip lain):
    python ../experiments/fair_compare_v1_v2.py
"""
import os
import sys

import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

from consistency import fold_consistency                           # noqa: E402
from embed import align_to_subset, assert_fold_unchanged, load_embeddings  # noqa: E402
from features import concat_features, l2norm                        # noqa: E402
from heads import make_head                                         # noqa: E402
from metrics import per_class_f1                                    # noqa: E402
from oof import assemble_oof, validate_oof                          # noqa: E402
from probe_cv import run_probe_cv                                   # noqa: E402
from resumable import run_or_load                                   # noqa: E402

HEADS = ["linear", "mlp", "lgbm"]
COMBOS = ["siglip2so400m", "concat_siglip2b256_siglip1b256_l2"]


def load_combo_features(combo: str, split: str) -> np.ndarray:
    """Identik head_grid_v3.py (kandidat sama) -- tidak diimpor lintas
    experiments/ (lihat catatan di resumable.py: reuse lintas experiment lewat
    src/, bukan impor antar-skrip), diduplikasi kecil & stabil."""
    if combo == "siglip2so400m":
        return l2norm(load_embeddings("siglip2so400m", split)[0])
    if combo == "concat_siglip2b256_siglip1b256_l2":
        blocks = [load_embeddings(n, split)[0] for n in ("siglip2b256", "siglip1b256")]
        return concat_features(blocks)
    raise KeyError(f"combo '{combo}' tidak dikenal. Pilihan: {COMBOS}")


def run_adil(X_v1: np.ndarray, folds_v1: pd.DataFrame, X_v2: np.ndarray,
            folds_v2: pd.DataFrame, head_name: str, class_weight="balanced",
            seed: int = 42) -> np.ndarray:
    """(ii) Adil: fit di baris v2 (fold != f), evaluasi di val-fold v1 UTUH
    (fold == f, termasuk baris yang di-drop dari v2). Tidak ada leakage: baris
    v2 dgn fold==f otomatis TIDAK masuk train (train_mask pakai fold v2 != f),
    dan fold assignment sampel yang bertahan identik v1<->v2 (assert_fold_unchanged
    sudah menegakkan ini di pemanggil)."""
    y_v1 = folds_v1["label"].to_numpy()
    y_v2 = folds_v2["label"].to_numpy()
    fold_v1 = folds_v1["fold"].to_numpy()
    fold_v2 = folds_v2["fold"].to_numpy()

    fold_probs = {}
    for f in sorted(folds_v1["fold"].unique()):
        train_mask = fold_v2 != f
        val_mask = fold_v1 == f

        head = make_head(head_name, seed=seed, class_weight=class_weight)
        head.fit(X_v2[train_mask], y_v2[train_mask])
        probs = head.predict_proba(X_v1[val_mask])

        val_idx = folds_v1.index[val_mask].to_numpy()
        fold_probs[int(f)] = (val_idx, probs)

    oof = assemble_oof(fold_probs, n_rows=len(folds_v1))
    validate_oof(oof, folds_v1)
    return oof


def build_comparison_row(head_name: str, combo: str, c_baseline: dict,
                         c_naif: dict, c_adil: dict,
                         pcf1_baseline, pcf1_naif, pcf1_adil) -> dict:
    """Pure function (tidak menyentuh IO) -- dites dgn fold_consistency() mock,
    sama pola dgn tta_compare.build_row."""
    return {
        "combo": combo, "head": head_name,
        "mean_baseline_v1": c_baseline["mean"], "min_baseline_v1": c_baseline["min"],
        "mean_naif_v2": c_naif["mean"], "min_naif_v2": c_naif["min"],
        "mean_adil_v2": c_adil["mean"], "min_adil_v2": c_adil["min"],
        "delta_naif_minus_adil": c_naif["mean"] - c_adil["mean"],
        "delta_adil_minus_baseline": c_adil["mean"] - c_baseline["mean"],
        "f1_electronic_baseline_v1": float(pcf1_baseline[1]),
        "f1_electronic_naif_v2": float(pcf1_naif[1]),
        "f1_electronic_adil_v2": float(pcf1_adil[1]),
    }


def compare_one(X_v1: np.ndarray, folds_v1: pd.DataFrame, X_v2: np.ndarray,
                folds_v2: pd.DataFrame, head_name: str, combo: str,
                out_dir: str, class_weight="balanced", seed: int = 42) -> dict:
    """Hitung (atau load cache) baseline/naif/adil untuk satu (combo, head),
    kembalikan satu baris perbandingan siap masuk tabel keputusan."""

    def _compute_baseline():
        oof, _ = run_probe_cv(X_v1, folds_v1, head_name, class_weight=class_weight, seed=seed)
        return oof, {"combo": combo, "head": head_name, "variant": "baseline_v1"}

    def _compute_naif():
        oof, _ = run_probe_cv(X_v2, folds_v2, head_name, class_weight=class_weight, seed=seed)
        return oof, {"combo": combo, "head": head_name, "variant": "naif_v2"}

    def _compute_adil():
        oof = run_adil(X_v1, folds_v1, X_v2, folds_v2, head_name,
                       class_weight=class_weight, seed=seed)
        return oof, {"combo": combo, "head": head_name, "variant": "adil_v2"}

    oof_baseline, _ = run_or_load(out_dir, f"faircmp_{combo}_{head_name}_baseline_v1", _compute_baseline)
    oof_naif, _ = run_or_load(out_dir, f"faircmp_{combo}_{head_name}_naif_v2", _compute_naif)
    oof_adil, _ = run_or_load(out_dir, f"faircmp_{combo}_{head_name}_adil_v2", _compute_adil)

    c_baseline = fold_consistency(oof_baseline, folds_v1)
    c_naif = fold_consistency(oof_naif, folds_v2)
    c_adil = fold_consistency(oof_adil, folds_v1)

    y_v1 = folds_v1["label"].to_numpy()
    y_v2 = folds_v2["label"].to_numpy()
    pcf1_baseline = per_class_f1(y_v1, oof_baseline.argmax(axis=1))
    pcf1_naif = per_class_f1(y_v2, oof_naif.argmax(axis=1))
    pcf1_adil = per_class_f1(y_v1, oof_adil.argmax(axis=1))

    return build_comparison_row(head_name, combo, c_baseline, c_naif, c_adil,
                                pcf1_baseline, pcf1_naif, pcf1_adil)


def run_fair_compare_grid(embeddings_v1: dict, folds_v1: pd.DataFrame,
                          embeddings_v2: dict, folds_v2: pd.DataFrame,
                          out_dir: str, combos: list = None, heads: list = None,
                          seed: int = 42) -> pd.DataFrame:
    combos = COMBOS if combos is None else combos
    heads = HEADS if heads is None else heads

    rows = [
        compare_one(embeddings_v1[combo], folds_v1, embeddings_v2[combo], folds_v2,
                   head_name, combo, out_dir, seed=seed)
        for combo in combos
        for head_name in heads
    ]
    return pd.DataFrame(rows)


def main() -> None:
    from config import CFG   # Drive-path coupling -- ditunda ke sini

    folds_v1 = pd.read_csv(CFG.folds_csv)
    folds_v2 = pd.read_csv(CFG.folds_v2_csv)
    assert_fold_unchanged(folds_v1, folds_v2)

    embeddings_v1 = {c: load_combo_features(c, "train") for c in COMBOS}
    embeddings_v2 = {c: align_to_subset(embeddings_v1[c], folds_v1, folds_v2) for c in COMBOS}

    out_dir = CFG.save_dir
    df = run_fair_compare_grid(embeddings_v1, folds_v1, embeddings_v2, folds_v2,
                               out_dir, seed=CFG.seed)

    repo_results = os.path.join(os.path.dirname(__file__), "..", "results", "fair_compare_v1_v2.csv")
    os.makedirs(os.path.dirname(repo_results), exist_ok=True)
    df.to_csv(repo_results, index=False)
    print(f"perbandingan tersimpan: {repo_results}")

    try:
        drive_results = os.path.join(out_dir, "fair_compare_v1_v2.csv")
        os.makedirs(os.path.dirname(drive_results), exist_ok=True)
        df.to_csv(drive_results, index=False)
    except OSError:
        pass

    print(df.to_string(index=False))
    print("\ndelta_naif_minus_adil besar -> banyak 'kenaikan' v2 cuma karena eval lebih mudah.")
    print("delta_adil_minus_baseline > 0 -> cleaning benar-benar membantu di kesulitan yang sama.")


if __name__ == "__main__":
    main()

"""head_grid_v3.py — Task 2 (TRACK_B_ARAHAN_V3.md): grid {kombinasi fitur} x
{head} x {folds_version}, resumable, output results/head_grid_v3.csv dengan
per-class F1 Electronic WAJIB ADA di tiap baris.

Kenapa per-class F1 wajib (B2): KNeighborsClassifier tidak menerima
class_weight -- sejak pivot ke kNN, penyeimbang kelas Electronic (14.9% data)
hilang tanpa pengganti. Head baru (A1: linear/mlp/lgbm) semuanya bisa
class_weight, jadi tiap kandidat WAJIB dibandingkan lewat F1 per kelas, bukan
cuma Macro-F1 agregat -- kalau tidak, kita menebak di mana sisa poinnya.

Kandidat (A1 + A3): {siglip2so400m, concat_siglip2b256_siglip1b256_l2} x
{linear, mlp, lgbm, ensemble}. Ensemble = rata-rata probabilitas 3 head
tunggal (B7), dihitung SETELAH ketiganya selesai -- bukan head yang di-fit
sendiri.

Dijalankan di folds.csv (v1) DAN folds_v2.csv (A4). Angka v2 di sini NAIF
(train+eval keduanya di v2) -- untuk perbandingan yang FAIR (train v2, eval
val fold v1 utuh) lihat fair_compare_v1_v2.py (B3), jangan disamakan.

Resumable: skip-if-exists per (combo, head, folds_version) -- Colab free tier
memutus sesi idle, jangan hitung ulang dari nol tiap kali runtime putus.

Jalankan dari track_b/src/ (sama seperti skrip lain):
    python ../experiments/head_grid_v3.py
"""
import os
import sys

import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

from consistency import fold_consistency               # noqa: E402
from embed import align_to_subset, assert_fold_unchanged, load_embeddings  # noqa: E402
from features import concat_features, l2norm            # noqa: E402
from metrics import per_class_f1                        # noqa: E402
from probe_cv import run_probe_cv                        # noqa: E402
from resumable import run_or_load as _run_or_load        # noqa: E402

SINGLE_HEADS = ["linear", "mlp", "lgbm"]
COMBOS = ["siglip2so400m", "concat_siglip2b256_siglip1b256_l2"]

RESULT_COLUMNS = ["combo", "head", "folds_version", "mean", "min", "std",
                  "f1_recyclable", "f1_electronic", "f1_organic"]


def load_combo_features(combo: str, split: str) -> np.ndarray:
    """Nama kombinasi -> fitur siap-pakai (sudah L2-norm). Kandidat baru
    didaftarkan di sini secara eksplisit -- salah ketik nama = KeyError
    langsung, sama semangatnya dengan heads.make_head."""
    if combo == "siglip2so400m":
        return l2norm(load_embeddings("siglip2so400m", split)[0])
    if combo == "concat_siglip2b256_siglip1b256_l2":
        blocks = [load_embeddings(n, split)[0] for n in ("siglip2b256", "siglip1b256")]
        return concat_features(blocks)          # L2-norm per-block sudah di dalamnya (B6)
    raise KeyError(f"combo '{combo}' tidak dikenal. Pilihan: {COMBOS}")


def oof_stem(combo: str, head: str, version: int) -> str:
    return f"oof_{combo}_{head}_v{version}"


def compute_candidate(X: np.ndarray, folds_df: pd.DataFrame, head_name: str,
                      combo: str, version: int, seed: int = 42) -> tuple:
    """Satu (combo, head, folds_version): 5-fold CV lewat probe_cv (dipakai
    ulang apa adanya, tidak ditulis ulang), rakit meta dengan per-class F1
    WAJIB ADA (B2)."""
    oof, _ = run_probe_cv(X, folds_df, head_name, class_weight="balanced", seed=seed)
    meta = _meta_from_oof(oof, folds_df, combo, head_name, version, seed, extra={"class_weight": "balanced"})
    return oof, meta


def ensemble_oof(oofs: list) -> np.ndarray:
    """B7: rata-rata probabilitas antar head -- diversity gratis, CPU saja."""
    assert len(oofs) >= 2, "ensemble butuh minimal 2 head"
    stacked = np.stack(oofs, axis=0)
    assert not np.isnan(stacked).any(), "salah satu OOF sumber ensemble punya NaN"
    return stacked.mean(axis=0)


def _meta_from_oof(oof: np.ndarray, folds_df: pd.DataFrame, combo: str, head_name: str,
                   version: int, seed: int, extra: dict = None) -> dict:
    c = fold_consistency(oof, folds_df)
    y = folds_df["label"].to_numpy()
    pcf1 = per_class_f1(y, oof.argmax(axis=1))
    meta = {
        "combo": combo, "head": head_name, "folds_version": version,
        "cv_mean": c["mean"], "cv_min": c["min"], "cv_std": c["std"],
        "per_fold": c["per_fold"],
        "f1_recyclable": float(pcf1[0]), "f1_electronic": float(pcf1[1]),
        "f1_organic": float(pcf1[2]),
        "n": int(len(folds_df)), "seed": seed,
    }
    if extra:
        meta.update(extra)
    return meta


def _row_from_meta(meta: dict) -> dict:
    return {
        "combo": meta["combo"], "head": meta["head"], "folds_version": meta["folds_version"],
        "mean": meta["cv_mean"], "min": meta["cv_min"], "std": meta["cv_std"],
        "f1_recyclable": meta["f1_recyclable"], "f1_electronic": meta["f1_electronic"],
        "f1_organic": meta["f1_organic"],
    }


def run_or_load(out_dir: str, combo: str, head_name: str, version: int, compute_fn) -> tuple:
    """Wrapper tipis di atas resumable.run_or_load dengan skema penamaan
    kontrak D: oof_{combo}_{head}_v{1|2}(.npy|_meta.json)."""
    return _run_or_load(out_dir, oof_stem(combo, head_name, version), compute_fn)


def run_grid(embeddings_train: dict, folds_by_version: dict, out_dir: str,
            combos: list = None, single_heads: list = None, seed: int = 42) -> pd.DataFrame:
    """Orchestrator murni-logika: `embeddings_train` = {combo_name: X} SUDAH
    di-load (testable tanpa Drive), `folds_by_version` = {1: folds_v1_df,
    2: folds_v2_df, ...}. IO (load_embeddings/CFG) tetap di main() -- fungsi
    ini dipanggil dari sana dengan data yang sudah dimuat."""
    combos = COMBOS if combos is None else combos
    single_heads = SINGLE_HEADS if single_heads is None else single_heads
    rows = []

    for version, folds_df in folds_by_version.items():
        for combo in combos:
            X = embeddings_train[combo]
            assert X.shape[0] == len(folds_df), \
                f"{combo} v{version}: X {X.shape[0]} baris != folds {len(folds_df)} baris"

            oofs_in_order = []
            for head_name in single_heads:
                def _compute(h=head_name):
                    return compute_candidate(X, folds_df, h, combo, version, seed)

                oof, meta = run_or_load(out_dir, combo, head_name, version, _compute)
                oofs_in_order.append(oof)
                rows.append(_row_from_meta(meta))

            def _compute_ensemble(_oofs=list(oofs_in_order)):
                oof = ensemble_oof(_oofs)
                meta = _meta_from_oof(oof, folds_df, combo, "ensemble", version, seed,
                                      extra={"ensemble_of": single_heads})
                return oof, meta

            _, meta = run_or_load(out_dir, combo, "ensemble", version, _compute_ensemble)
            rows.append(_row_from_meta(meta))

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def main() -> None:
    from config import CFG   # Drive-path coupling -- ditunda ke sini, bukan top-level

    folds_v1 = pd.read_csv(CFG.folds_csv)
    folds_v2 = pd.read_csv(CFG.folds_v2_csv)
    assert_fold_unchanged(folds_v1, folds_v2)   # guard D: sampel bertahan tak boleh pindah fold

    embeddings_v1 = {c: load_combo_features(c, "train") for c in COMBOS}
    embeddings_v2 = {c: align_to_subset(embeddings_v1[c], folds_v1, folds_v2) for c in COMBOS}

    out_dir = CFG.save_dir
    df_v1 = run_grid(embeddings_v1, {1: folds_v1}, out_dir, seed=CFG.seed)
    df_v2 = run_grid(embeddings_v2, {2: folds_v2}, out_dir, seed=CFG.seed)
    df = pd.concat([df_v1, df_v2], ignore_index=True).sort_values(
        ["folds_version", "mean"], ascending=[True, False])

    repo_results = os.path.join(os.path.dirname(__file__), "..", "results", "head_grid_v3.csv")
    os.makedirs(os.path.dirname(repo_results), exist_ok=True)
    df.to_csv(repo_results, index=False)
    print(f"tabel keputusan tersimpan: {repo_results}")

    try:
        drive_results = os.path.join(out_dir, "head_grid_v3.csv")
        os.makedirs(os.path.dirname(drive_results), exist_ok=True)
        df.to_csv(drive_results, index=False)
    except OSError:
        pass

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

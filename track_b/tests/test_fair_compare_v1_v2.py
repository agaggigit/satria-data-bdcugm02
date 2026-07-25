"""Test CPU-only untuk fair_compare_v1_v2.py (Task 3, TRACK_B_ARAHAN_V3.md B3).
Data sintetis -- tidak menyentuh Drive/embedding asli. sklearn dipakai
sungguhan, sama pola dengan test_probe_cv.py / test_head_grid_v3.py."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from fair_compare_v1_v2 import (build_comparison_row, compare_one,
                                run_adil, run_fair_compare_grid)

rng = np.random.default_rng(42)


def _separable(n=90, d=6, seed=42):
    r = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], n // 3)
    centers = r.normal(size=(3, d)) * 6
    X = centers[y] + r.normal(scale=0.3, size=(n, d))
    return X, y, centers


def _folds_df(y, n_folds=5):
    fold = np.tile(np.arange(n_folds), len(y) // n_folds + 1)[:len(y)]
    return pd.DataFrame({
        "filepath": [f"img{i}.jpg" for i in range(len(y))],
        "label": y,
        "fold": fold,
    })


def _easy_and_hard(seed=42, d=6):
    """Titik 'hard' ditaruh TEPAT di tengah dua centroid kelas LAIN -- ambigu
    by construction, apa pun modelnya. Simulasi sampel yang dibuang Track A
    saat cleaning (mis-label / ambigu secara visual)."""
    X_easy, y_easy, centers = _separable(n=90, d=d, seed=seed)
    r = np.random.default_rng(seed + 1)

    n_hard = 15
    hard_labels = np.tile([0, 1, 2], n_hard // 3)
    X_hard = []
    for lbl in hard_labels:
        other = [c for c in range(3) if c != lbl]
        mid = (centers[other[0]] + centers[other[1]]) / 2
        X_hard.append(mid + r.normal(scale=0.1, size=d))
    X_hard = np.array(X_hard)

    X = np.vstack([X_easy, X_hard])
    y = np.concatenate([y_easy, hard_labels])
    is_hard = np.concatenate([np.zeros(len(y_easy), bool), np.ones(n_hard, bool)])
    return X, y, is_hard


# --- run_adil: tidak boleh bocor, dan harus menutupi seluruh v1 tepat sekali ---

def test_run_adil_covers_every_v1_row_exactly_once():
    X, y, _ = _separable(n=90)
    folds_v1 = _folds_df(y)
    keep = np.sort(np.random.default_rng(7).choice(90, 60, replace=False))
    folds_v2 = folds_v1.iloc[keep].reset_index(drop=True)
    X_v2 = X[keep]

    oof = run_adil(X, folds_v1, X_v2, folds_v2, "linear")

    assert oof.shape == (90, 3)
    assert not np.isnan(oof).any()
    assert np.allclose(oof.sum(axis=1), 1.0, atol=1e-5)


def test_run_adil_random_noise_scores_near_chance():
    """Anti-leakage: fitur acak tanpa sinyal -> skor rendah, bukan mustahil tinggi."""
    _, y, _ = _separable(n=90)
    folds_v1 = _folds_df(y)
    keep = np.sort(np.random.default_rng(7).choice(90, 60, replace=False))
    folds_v2 = folds_v1.iloc[keep].reset_index(drop=True)

    X_noise = rng.normal(size=(90, 6))
    X_noise_v2 = X_noise[keep]

    oof = run_adil(X_noise, folds_v1, X_noise_v2, folds_v2, "linear")
    acc = (oof.argmax(axis=1) == y).mean()
    assert acc < 0.6, "fitur acak tapi akurasi tinggi -- run_adil bocor"


def test_run_adil_training_rows_never_overlap_validation_fold():
    """Bukti langsung anti-leakage: untuk tiap fold f, tidak ada baris v2 dgn
    fold==f yang ikut dipakai fit model yang memvalidasi fold f."""
    X, y, _ = _separable(n=90)
    folds_v1 = _folds_df(y)
    keep = np.sort(np.random.default_rng(7).choice(90, 60, replace=False))
    folds_v2 = folds_v1.iloc[keep].reset_index(drop=True)
    X_v2 = X[keep]

    for f in sorted(folds_v1["fold"].unique()):
        train_idx_v2 = set(np.where(folds_v2["fold"].to_numpy() != f)[0])
        val_fp_v1 = set(folds_v1.loc[folds_v1["fold"] == f, "filepath"])
        train_fp_v2 = set(folds_v2.iloc[sorted(train_idx_v2)]["filepath"])
        assert not (train_fp_v2 & val_fp_v1), f"fold {f}: ada overlap train/val -- leakage"


# --- Mekanisme inti B3: naif inflasi karena sampel sulit dibuang dari eval ---

def test_fair_comparison_reveals_naif_inflation_from_dropped_hard_samples(tmp_path):
    X, y, is_hard = _easy_and_hard()
    folds_v1 = _folds_df(y)
    keep = ~is_hard   # v2 = buang semua sampel "hard" (simulasi cleaning)
    folds_v2 = folds_v1[keep].reset_index(drop=True)
    X_v2 = X[keep]

    row = compare_one(X, folds_v1, X_v2, folds_v2, "linear", "combo_x", str(tmp_path))

    assert row["mean_naif_v2"] > row["mean_adil_v2"] + 0.05, (
        f"naif={row['mean_naif_v2']:.4f} adil={row['mean_adil_v2']:.4f} -- "
        f"naif (eval TANPA sampel hard) seharusnya jauh lebih tinggi dari adil "
        f"(eval DENGAN sampel hard yang sama seperti v1); kalau tidak, mekanisme "
        f"inflasi yang diperingatkan B3 tidak tertangkap"
    )
    assert row["delta_naif_minus_adil"] > 0.05


def test_fair_comparison_adil_and_baseline_share_the_same_eval_difficulty(tmp_path):
    """adil dan baseline dua-duanya dievaluasi di val fold v1 UTUH -- beda
    keduanya murni soal data TRAIN (v2 bersih vs v1 apa adanya), bukan soal
    eval set yang beda kesulitan."""
    X, y, is_hard = _easy_and_hard()
    folds_v1 = _folds_df(y)
    keep = ~is_hard
    folds_v2 = folds_v1[keep].reset_index(drop=True)
    X_v2 = X[keep]

    row = compare_one(X, folds_v1, X_v2, folds_v2, "linear", "combo_x", str(tmp_path))

    # delta_adil_minus_baseline harus proporsi kecil dibanding delta_naif_minus_adil
    # -- gain data-cleaning yang sungguhan jauh lebih kecil dari inflasi eval semu.
    assert abs(row["delta_adil_minus_baseline"]) < row["delta_naif_minus_adil"]


# --- build_comparison_row: pure function, kolom & delta benar ---

def _cons(mean, mn):
    return {"mean": mean, "min": mn, "std": 0.0, "per_fold": []}


def test_build_comparison_row_computes_deltas_correctly():
    row = build_comparison_row(
        "linear", "combo_x",
        c_baseline=_cons(0.90, 0.88),
        c_naif=_cons(0.95, 0.93),
        c_adil=_cons(0.91, 0.89),
        pcf1_baseline=[0.8, 0.7, 0.9],
        pcf1_naif=[0.85, 0.8, 0.92],
        pcf1_adil=[0.81, 0.72, 0.90],
    )
    assert abs(row["delta_naif_minus_adil"] - (0.95 - 0.91)) < 1e-9
    assert abs(row["delta_adil_minus_baseline"] - (0.91 - 0.90)) < 1e-9
    assert row["f1_electronic_baseline_v1"] == 0.7
    assert row["f1_electronic_naif_v2"] == 0.8
    assert row["f1_electronic_adil_v2"] == 0.72


def test_build_comparison_row_has_all_required_columns():
    row = build_comparison_row(
        "mlp", "combo_x", _cons(0.9, 0.88), _cons(0.9, 0.88), _cons(0.9, 0.88),
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    required = {"combo", "head", "mean_baseline_v1", "min_baseline_v1",
               "mean_naif_v2", "min_naif_v2", "mean_adil_v2", "min_adil_v2",
               "delta_naif_minus_adil", "delta_adil_minus_baseline",
               "f1_electronic_baseline_v1", "f1_electronic_naif_v2",
               "f1_electronic_adil_v2"}
    assert required <= set(row)


# --- compare_one: resumable ---

def test_compare_one_is_resumable_across_calls(tmp_path, monkeypatch):
    import fair_compare_v1_v2 as mod

    X, y, _ = _separable(n=90)
    folds_v1 = _folds_df(y)
    keep = np.sort(np.random.default_rng(7).choice(90, 60, replace=False))
    folds_v2 = folds_v1.iloc[keep].reset_index(drop=True)
    X_v2 = X[keep]

    calls = {"adil": 0}
    original_run_adil = mod.run_adil

    def counting_run_adil(*args, **kwargs):
        calls["adil"] += 1
        return original_run_adil(*args, **kwargs)

    monkeypatch.setattr(mod, "run_adil", counting_run_adil)

    row1 = compare_one(X, folds_v1, X_v2, folds_v2, "linear", "combo_x", str(tmp_path))
    assert calls["adil"] == 1
    row2 = compare_one(X, folds_v1, X_v2, folds_v2, "linear", "combo_x", str(tmp_path))
    assert calls["adil"] == 1, "compare_one kedua menghitung ulang 'adil' -- resumability rusak"
    assert row1 == row2


# --- run_fair_compare_grid: orkestrasi, satu baris per (combo, head) ---

def test_run_fair_compare_grid_has_one_row_per_combo_and_head(tmp_path):
    X, y, _ = _separable(n=90)
    folds_v1 = _folds_df(y)
    keep = np.sort(np.random.default_rng(7).choice(90, 60, replace=False))
    folds_v2 = folds_v1.iloc[keep].reset_index(drop=True)
    X_v2 = X[keep]

    df = run_fair_compare_grid(
        {"combo_a": X}, folds_v1, {"combo_a": X_v2}, folds_v2, str(tmp_path),
        combos=["combo_a"], heads=["linear", "mlp"])

    assert len(df) == 2
    assert set(df["head"]) == {"linear", "mlp"}
    assert (df["combo"] == "combo_a").all()

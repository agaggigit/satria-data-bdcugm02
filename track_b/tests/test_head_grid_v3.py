"""Test CPU-only untuk head_grid_v3.py (Task 2, TRACK_B_ARAHAN_V3.md). Data
sintetis kecil -- tidak menyentuh Drive/embedding asli. sklearn dipakai
sungguhan (bukan mock) untuk membuktikan plumbing CV-nya benar, sama seperti
test_probe_cv.py."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

# experiments/ bukan package & tidak di sys.path lewat conftest -> tambah manual.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import embed
from head_grid_v3 import (RESULT_COLUMNS, compute_candidate, ensemble_oof,
                          load_combo_features, oof_stem, run_grid, run_or_load)

rng = np.random.default_rng(42)


def _separable(n=90, d=8, seed=42):
    r = np.random.default_rng(seed)
    y = np.repeat([0, 1, 2], n // 3)
    centers = r.normal(size=(3, d)) * 5
    X = centers[y] + r.normal(scale=0.3, size=(n, d))
    return X, y


def _folds_df(y, n_folds=5):
    fold = np.tile(np.arange(n_folds), len(y) // n_folds + 1)[:len(y)]
    return pd.DataFrame({
        "filepath": [f"img{i}.jpg" for i in range(len(y))],
        "label": y,
        "fold": fold,
    })


# --- load_combo_features: nama kombinasi -> fitur L2-norm siap pakai ---

def _save_emb(name, split, dim, n=20, checkpoint="ckpt/fake"):
    e = np.random.default_rng(0).normal(size=(n, dim)).astype(np.float32) + 5.0  # jauh dari 0
    embed.save_embeddings(e, name, split, {"checkpoint": checkpoint, "flips": [], "seed": 42})
    return e


def test_load_combo_features_single_backbone_is_l2_normalized(tmp_path, monkeypatch):
    monkeypatch.setattr("embed.EMB_DIR", tmp_path)
    _save_emb("siglip2so400m", "train", dim=6)

    X = load_combo_features("siglip2so400m", "train")

    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)


def test_load_combo_features_concat_has_correct_total_dim_and_is_l2_normalized_per_block(tmp_path, monkeypatch):
    monkeypatch.setattr("embed.EMB_DIR", tmp_path)
    _save_emb("siglip2b256", "train", dim=4)
    _save_emb("siglip1b256", "train", dim=5)

    X = load_combo_features("concat_siglip2b256_siglip1b256_l2", "train")

    assert X.shape[1] == 9   # 4 + 5
    block_a_norm = np.linalg.norm(X[:, :4], axis=1)
    block_b_norm = np.linalg.norm(X[:, 4:], axis=1)
    assert np.allclose(block_a_norm, 1.0, atol=1e-5)
    assert np.allclose(block_b_norm, 1.0, atol=1e-5)


def test_load_combo_features_rejects_unknown_combo_immediately():
    with pytest.raises(KeyError, match="tidak dikenal"):
        load_combo_features("combo_yang_tidak_ada", "train")


# --- run_or_load: resumable, skip-if-exists, compute_fn TIDAK dipanggil kalau cache ada ---

def test_run_or_load_calls_compute_fn_only_once_across_two_calls(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return np.full((5, 3), 1 / 3), {"combo": "c", "head": "h", "folds_version": 1,
                                        "cv_mean": 0.5, "cv_min": 0.5, "cv_std": 0.0,
                                        "per_fold": [0.5], "f1_recyclable": 0.5,
                                        "f1_electronic": 0.5, "f1_organic": 0.5, "n": 5, "seed": 42}

    run_or_load(str(tmp_path), "c", "h", 1, compute)
    run_or_load(str(tmp_path), "c", "h", 1, compute)   # panggilan ke-2 -- harus dari cache

    assert len(calls) == 1, "compute_fn dipanggil lagi padahal cache sudah ada -- resumability rusak"


def test_run_or_load_returns_cached_marker_value_unchanged(tmp_path):
    """Bukti langsung skip beneran (bukan cuma 'kebetulan sama'): seed nilai
    PENANDA yang mustahil dihasilkan compute_fn, lalu pastikan itu yang balik."""
    stem = oof_stem("c", "h", 1)
    marker = np.full((4, 3), -1.0)
    np.save(os.path.join(tmp_path, f"{stem}.npy"), marker)
    with open(os.path.join(tmp_path, f"{stem}_meta.json"), "w") as f:
        json.dump({"combo": "c", "head": "h", "folds_version": 1, "cv_mean": -1,
                  "cv_min": -1, "cv_std": -1, "per_fold": [], "f1_recyclable": -1,
                  "f1_electronic": -1, "f1_organic": -1, "n": 4, "seed": 42}, f)

    def compute():
        raise AssertionError("compute_fn TIDAK BOLEH dipanggil -- cache lengkap sudah ada")

    oof, meta = run_or_load(str(tmp_path), "c", "h", 1, compute)

    assert np.allclose(oof, -1.0)
    assert meta["cv_mean"] == -1


def test_run_or_load_computes_and_persists_when_cache_missing(tmp_path):
    def compute():
        return np.full((3, 3), 1 / 3), {"combo": "c", "head": "h", "folds_version": 1,
                                        "cv_mean": 0.77, "cv_min": 0.7, "cv_std": 0.01,
                                        "per_fold": [0.7, 0.8], "f1_recyclable": 0.7,
                                        "f1_electronic": 0.6, "f1_organic": 0.8, "n": 3, "seed": 42}

    oof, meta = run_or_load(str(tmp_path), "c", "h", 1, compute)

    assert meta["cv_mean"] == 0.77
    stem = oof_stem("c", "h", 1)
    assert os.path.exists(os.path.join(tmp_path, f"{stem}.npy"))
    assert os.path.exists(os.path.join(tmp_path, f"{stem}_meta.json"))


# --- ensemble_oof: rata-rata probabilitas antar head (B7) ---

def test_ensemble_oof_averages_elementwise():
    a = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    b = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3]])
    ens = ensemble_oof([a, b])
    assert np.allclose(ens, (a + b) / 2)


def test_ensemble_oof_rejects_nan_source():
    a = np.array([[0.9, 0.05, 0.05]])
    b = np.array([[np.nan, 0.5, 0.5]])
    with pytest.raises(AssertionError, match="NaN"):
        ensemble_oof([a, b])


def test_ensemble_oof_requires_at_least_two():
    with pytest.raises(AssertionError, match="minimal 2"):
        ensemble_oof([np.zeros((3, 3))])


# --- compute_candidate: per-class F1 WAJIB ADA (B2) ---

@pytest.mark.parametrize("head_name", ["linear", "mlp", "lgbm"])
def test_compute_candidate_meta_has_per_class_f1_electronic(head_name):
    X, y = _separable()
    folds_df = _folds_df(y)
    oof, meta = compute_candidate(X, folds_df, head_name, "combo_x", version=1, seed=42)

    assert oof.shape == (len(y), 3)
    for key in ("f1_recyclable", "f1_electronic", "f1_organic"):
        assert key in meta
        assert 0.0 <= meta[key] <= 1.0
    assert meta["cv_mean"] > 0.9, "data jelas terpisah tapi CV rendah -- plumbing salah"


# --- run_grid: orkestrasi penuh, kolom benar, resumable, folds_version benar ---

def test_run_grid_produces_expected_columns_and_row_count(tmp_path):
    X, y = _separable(n=90)
    folds_df = _folds_df(y)

    df = run_grid({"combo_a": X}, {1: folds_df}, str(tmp_path), combos=["combo_a"])

    assert list(df.columns) == RESULT_COLUMNS
    # 3 head tunggal + 1 ensemble, untuk 1 combo x 1 folds_version
    assert len(df) == 4
    assert set(df["head"]) == {"linear", "mlp", "lgbm", "ensemble"}
    assert (df["folds_version"] == 1).all()


def test_run_grid_reports_per_class_f1_electronic_for_every_row(tmp_path):
    X, y = _separable(n=90)
    folds_df = _folds_df(y)

    df = run_grid({"combo_a": X}, {1: folds_df}, str(tmp_path), combos=["combo_a"])

    assert df["f1_electronic"].notna().all()
    assert (df["f1_electronic"] >= 0).all() and (df["f1_electronic"] <= 1).all()


def test_run_grid_separates_multiple_folds_versions(tmp_path):
    X, y = _separable(n=90)
    folds_v1 = _folds_df(y)

    # Subset acak (BUKAN potongan posisional) -- data sintetis tersusun per
    # blok kelas, jadi potongan posisional bisa menghapus 1 kelas seluruhnya.
    # Mirip kondisi asli: Track A cuma drop ~2.3% sampel, bukan hapus kelas.
    keep = np.sort(rng.choice(len(y), size=60, replace=False))
    folds_v2 = folds_v1.iloc[keep].reset_index(drop=True)
    X_v2 = X[keep]

    df = run_grid(
        {"combo_a": X}, {1: folds_v1}, str(tmp_path), combos=["combo_a"])
    df2 = run_grid(
        {"combo_a": X_v2}, {2: folds_v2}, str(tmp_path), combos=["combo_a"])
    df_all = pd.concat([df, df2], ignore_index=True)

    assert set(df_all["folds_version"]) == {1, 2}
    assert len(df_all) == 8   # 4 baris x 2 versi


def test_run_grid_is_resumable_across_calls(tmp_path, monkeypatch):
    """Panggilan kedua ke run_grid dgn out_dir sama TIDAK BOLEH menghitung ulang
    -- dibuktikan dgn menghitung berapa kali compute_candidate benar-benar
    dipanggil lintas dua run_grid() terpisah."""
    import head_grid_v3

    X, y = _separable(n=90)
    folds_df = _folds_df(y)

    calls = []
    original = head_grid_v3.compute_candidate

    def counting_compute_candidate(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(head_grid_v3, "compute_candidate", counting_compute_candidate)

    df1 = run_grid({"combo_a": X}, {1: folds_df}, str(tmp_path), combos=["combo_a"])
    n_calls_after_first = len(calls)
    assert n_calls_after_first == 3   # 3 head tunggal (ensemble tidak lewat compute_candidate)

    df2 = run_grid({"combo_a": X}, {1: folds_df}, str(tmp_path), combos=["combo_a"])

    assert len(calls) == n_calls_after_first, \
        "run_grid kedua memanggil compute_candidate lagi -- resumability rusak"
    pd.testing.assert_frame_equal(
        df1.reset_index(drop=True), df2.reset_index(drop=True))


def test_run_grid_random_noise_features_score_near_chance(tmp_path):
    """Sanity anti-leakage di level orkestrasi penuh (bukan cuma probe_cv
    sendirian): fitur acak tanpa sinyal -> CV harus rendah, bukan sempurna."""
    _, y = _separable(n=90)
    folds_df = _folds_df(y)
    X_noise = rng.normal(size=(90, 8))

    df = run_grid({"combo_a": X_noise}, {1: folds_df}, str(tmp_path), combos=["combo_a"])

    assert (df["mean"] < 0.6).all(), "fitur acak tapi CV tinggi -- OOF di run_grid bocor"

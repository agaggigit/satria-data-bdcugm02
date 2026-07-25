import json
import os

import numpy as np
import pandas as pd
import pytest

from handoff_v3 import handoff_winner, select_source_oof_path


def _folds_v1(n=10):
    return pd.DataFrame({
        "filepath": [f"img{i}.jpg" for i in range(n)],
        "label": [i % 3 for i in range(n)],
        "fold": [i % 5 for i in range(n)],
    })


def _write_oof(cache_dir, stem, oof, meta):
    np.save(os.path.join(cache_dir, f"{stem}.npy"), oof)
    with open(os.path.join(cache_dir, f"{stem}_meta.json"), "w") as f:
        json.dump(meta, f)


def _valid_oof(n=10, seed=0):
    r = np.random.default_rng(seed)
    raw = r.random((n, 3))
    return raw / raw.sum(axis=1, keepdims=True)


# --- select_source_oof_path: v1 langsung, v2 lewat varian 'adil' ---

def test_select_source_uses_head_grid_stem_for_v1(tmp_path):
    npy, meta = select_source_oof_path(str(tmp_path), "combo_x", "linear", 1)
    assert npy.endswith("oof_combo_x_linear_v1.npy")
    assert meta.endswith("oof_combo_x_linear_v1_meta.json")


def test_select_source_uses_fair_compare_adil_stem_for_v2(tmp_path):
    npy, meta = select_source_oof_path(str(tmp_path), "combo_x", "linear", 2)
    assert npy.endswith("faircmp_combo_x_linear_adil_v2.npy")
    assert meta.endswith("faircmp_combo_x_linear_adil_v2_meta.json")


def test_select_source_rejects_invalid_version(tmp_path):
    with pytest.raises(ValueError, match="1 atau 2"):
        select_source_oof_path(str(tmp_path), "combo_x", "linear", 3)


# --- handoff_winner: sumber hilang, alignment, NaN, anti-overwrite, isi meta ---

def test_handoff_raises_when_source_missing(tmp_path):
    folds_v1 = _folds_v1()
    with pytest.raises(FileNotFoundError, match="tidak ada"):
        handoff_winner(str(tmp_path), "combo_x", "linear", 1, folds_v1, str(tmp_path / "out"))


def test_handoff_rejects_misaligned_oof(tmp_path):
    """Simulasi kesalahan: pakai OOF naif v2 (baris lebih sedikit) langsung
    tanpa lewat fair_compare -- harus ditolak, bukan diam-diam diserahkan."""
    folds_v1 = _folds_v1(n=10)
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_oof(cache, "oof_combo_x_linear_v1", _valid_oof(n=7), {"cv_mean": 0.9})  # 7 != 10

    with pytest.raises(AssertionError, match="TIDAK align"):
        handoff_winner(str(cache), "combo_x", "linear", 1, folds_v1, str(tmp_path / "out"))


def test_handoff_rejects_nan_oof(tmp_path):
    folds_v1 = _folds_v1(n=5)
    cache = tmp_path / "cache"
    cache.mkdir()
    oof = _valid_oof(n=5)
    oof[2, 0] = np.nan
    _write_oof(cache, "oof_combo_x_linear_v1", oof, {"cv_mean": 0.9})

    with pytest.raises(AssertionError, match="NaN"):
        handoff_winner(str(cache), "combo_x", "linear", 1, folds_v1, str(tmp_path / "out"))


def test_handoff_writes_oof_and_meta_with_correct_macro_f1(tmp_path):
    folds_v1 = pd.DataFrame({
        "filepath": [f"img{i}.jpg" for i in range(6)],
        "label": [0, 1, 2, 0, 1, 2],
        "fold": [0, 0, 1, 1, 2, 2],
    })
    cache = tmp_path / "cache"
    cache.mkdir()
    # OOF "sempurna" -- argmax selalu sama dengan label -> macro_f1 harus 1.0
    oof = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ], dtype=np.float64)
    _write_oof(cache, "oof_combo_x_linear_v1", oof, {"cv_mean": 0.95, "cv_min": 0.94, "seed": 42})

    out_dir = tmp_path / "out"
    meta = handoff_winner(str(cache), "combo_x", "linear", 1, folds_v1, str(out_dir))

    assert meta["oof_overall_macro_f1_argmax"] == 1.0
    assert meta["combo"] == "combo_x"
    assert meta["head"] == "linear"
    assert meta["folds_version_trained_on"] == 1
    assert os.path.exists(out_dir / "oof.npy")
    assert os.path.exists(out_dir / "oof_meta.json")
    loaded = np.load(out_dir / "oof.npy")
    assert np.allclose(loaded, oof)


def test_handoff_refuses_to_overwrite_existing_canonical_oof(tmp_path):
    folds_v1 = _folds_v1(n=6)
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_oof(cache, "oof_combo_x_linear_v1", _valid_oof(n=6), {"cv_mean": 0.9})

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "oof.npy").write_bytes(b"\x00")   # simulasi handoff lama (era ConvNeXt)

    with pytest.raises(FileExistsError, match="sudah ada"):
        handoff_winner(str(cache), "combo_x", "linear", 1, folds_v1, str(out_dir))


def test_handoff_allows_overwrite_when_explicitly_set(tmp_path):
    folds_v1 = _folds_v1(n=6)
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_oof(cache, "oof_combo_x_linear_v1", _valid_oof(n=6), {"cv_mean": 0.9})

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "oof.npy").write_bytes(b"\x00")

    meta = handoff_winner(str(cache), "combo_x", "linear", 1, folds_v1, str(out_dir),
                          allow_overwrite=True)
    assert meta["combo"] == "combo_x"


def test_handoff_v2_reads_from_fair_compare_adil_stem_not_raw_naif(tmp_path):
    """Pemenang folds_version=2 HARUS baca dari faircmp..._adil_v2 (align v1),
    BUKAN dari oof_..._v2 (naif, align v2 yang lebih pendek)."""
    folds_v1 = _folds_v1(n=8)
    cache = tmp_path / "cache"
    cache.mkdir()
    # Sengaja taruh oof_..._v2 (naif) dengan panjang SALAH (align v2, bukan v1)
    _write_oof(cache, "oof_combo_x_mlp_v2", _valid_oof(n=5), {"cv_mean": 0.99})
    # Yang benar: varian adil, align penuh ke v1 (8 baris)
    _write_oof(cache, "faircmp_combo_x_mlp_adil_v2", _valid_oof(n=8), {"cv_mean": 0.88})

    meta = handoff_winner(str(cache), "combo_x", "mlp", 2, folds_v1, str(tmp_path / "out"))
    assert meta["source_stem"] == "faircmp_combo_x_mlp_adil_v2"

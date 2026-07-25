import json
import os

import numpy as np
import pytest

from resumable import run_or_load


def test_computes_and_persists_when_cache_missing(tmp_path):
    def compute():
        return np.full((3, 2), 0.5), {"x": 1}

    arr, meta = run_or_load(str(tmp_path), "stem", compute)

    assert np.allclose(arr, 0.5)
    assert meta == {"x": 1}
    assert os.path.exists(os.path.join(tmp_path, "stem.npy"))
    assert os.path.exists(os.path.join(tmp_path, "stem_meta.json"))


def test_compute_fn_called_only_once_across_two_calls(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return np.zeros((2, 2)), {"n": len(calls)}

    run_or_load(str(tmp_path), "stem", compute)
    run_or_load(str(tmp_path), "stem", compute)

    assert len(calls) == 1


def test_returns_cached_marker_value_unchanged_and_never_calls_compute(tmp_path):
    np.save(os.path.join(tmp_path, "stem.npy"), np.full((2, 2), -1.0))
    with open(os.path.join(tmp_path, "stem_meta.json"), "w") as f:
        json.dump({"marker": True}, f)

    def compute():
        raise AssertionError("compute_fn TIDAK BOLEH dipanggil -- cache lengkap sudah ada")

    arr, meta = run_or_load(str(tmp_path), "stem", compute)

    assert np.allclose(arr, -1.0)
    assert meta == {"marker": True}


def test_npy_without_meta_is_not_considered_cached(tmp_path):
    """Proses yang mati di tengah jalan bisa meninggalkan .npy tanpa meta --
    itu HARUS dihitung ulang, bukan dikira sudah lengkap."""
    np.save(os.path.join(tmp_path, "stem.npy"), np.full((2, 2), -1.0))  # tanpa meta

    def compute():
        return np.full((2, 2), 7.0), {"recomputed": True}

    arr, meta = run_or_load(str(tmp_path), "stem", compute)

    assert np.allclose(arr, 7.0)
    assert meta == {"recomputed": True}


def test_different_stems_do_not_collide(tmp_path):
    def compute_a():
        return np.full((1, 1), 1.0), {"which": "a"}

    def compute_b():
        return np.full((1, 1), 2.0), {"which": "b"}

    arr_a, meta_a = run_or_load(str(tmp_path), "stem_a", compute_a)
    arr_b, meta_b = run_or_load(str(tmp_path), "stem_b", compute_b)

    assert arr_a[0, 0] == 1.0 and meta_a["which"] == "a"
    assert arr_b[0, 0] == 2.0 and meta_b["which"] == "b"

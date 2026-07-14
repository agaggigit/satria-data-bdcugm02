import json

import numpy as np
import pandas as pd
import pytest

from embed import assert_aligned, emb_path, load_embeddings, save_embeddings


@pytest.fixture
def folds_df():
    rows = [{"filepath": f"img{i}.jpg", "label": i % 3, "fold": i % 5} for i in range(20)]
    return pd.DataFrame(rows)


def _meta():
    return {"checkpoint": "google/siglip2-base-patch16-256", "dim": 8,
            "n_rows": 20, "flips": [], "seed": 42}


def test_save_and_load_roundtrip(tmp_path, monkeypatch, folds_df):
    monkeypatch.setattr("embed.EMB_DIR", tmp_path)
    emb = np.random.rand(20, 8).astype(np.float32)
    save_embeddings(emb, "siglip2b256", "train", _meta())

    loaded, meta = load_embeddings("siglip2b256", "train")
    assert np.allclose(loaded, emb)
    assert meta["checkpoint"] == "google/siglip2-base-patch16-256"


def test_save_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr("embed.EMB_DIR", tmp_path)
    emb = np.random.rand(20, 8).astype(np.float32)
    save_embeddings(emb, "siglip2b256", "train", _meta())
    with pytest.raises(FileExistsError, match="sudah ada"):
        save_embeddings(emb, "siglip2b256", "train", _meta())


def test_manifest_is_written_next_to_the_array(tmp_path, monkeypatch):
    monkeypatch.setattr("embed.EMB_DIR", tmp_path)
    save_embeddings(np.zeros((20, 8), np.float32), "siglip2b256", "train", _meta())
    manifest = json.loads((tmp_path / "siglip2b256_train.json").read_text())
    assert manifest["checkpoint"] == "google/siglip2-base-patch16-256"
    assert manifest["n_rows"] == 20


def test_assert_aligned_rejects_wrong_row_count(folds_df):
    with pytest.raises(AssertionError, match="baris"):
        assert_aligned(np.zeros((19, 8)), folds_df)   # 19 != 20


def test_assert_aligned_rejects_nan(folds_df):
    emb = np.zeros((20, 8))
    emb[3, 0] = np.nan
    with pytest.raises(AssertionError, match="NaN"):
        assert_aligned(emb, folds_df)


def test_assert_aligned_accepts_valid(folds_df):
    assert_aligned(np.random.rand(20, 8), folds_df)   # tidak raise

import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from embed import (assert_aligned, emb_path, extract_embeddings,
                   load_embeddings, save_embeddings)


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


# --- extract_embeddings: loop RAM-safe, diuji CPU-only tanpa unduh bobot ---
#
# Encoder palsu: processor membaca kanal R piksel (0,0) tiap gambar sebagai
# "identitas", model men-tile identitas itu jadi embedding [N, 4]. Dengan begitu
# baris output HARUS mencerminkan gambar input pada posisi yang sama -> ini yang
# menangkap kalau pre-alokasi / penulisan per-posisi merusak urutan (alignment).

class _FakeInputs(dict):
    def to(self, *args, **kwargs):
        return self


class _FakeProcessor:
    def __call__(self, images, return_tensors=None):
        ids = [float(np.asarray(im)[0, 0, 0]) for im in images]
        return _FakeInputs(pixel_values=torch.tensor(ids).view(-1, 1))


class _FakeModel:
    def get_image_features(self, pixel_values=None):
        return pixel_values.repeat(1, 4)      # D = 4


def _write_solid_pngs(dirpath, values):
    paths = []
    for i, v in enumerate(values):
        p = dirpath / f"img{i}.png"
        Image.new("RGB", (8, 8), color=(v, 0, 0)).save(p)
        paths.append(str(p))
    return paths


def test_extract_preserves_row_order_across_batch_boundaries(tmp_path):
    values = [10, 20, 30, 40, 50, 60, 70]          # 7 gambar, batch 3 -> 3+3+1
    paths = _write_solid_pngs(tmp_path, values)
    encoder = (_FakeModel(), _FakeProcessor())

    emb = extract_embeddings("fake", paths, device="cpu", batch=3,
                             encoder=encoder, log_ram=False)

    assert emb.shape == (7, 4)
    assert emb.dtype == np.float32
    # baris ke-i harus = identitas gambar ke-i (urutan terjaga lintas batch)
    assert np.allclose(emb[:, 0], values)
    assert not np.isnan(emb).any()


def test_extract_tta_averages_at_embedding_level(tmp_path):
    paths = _write_solid_pngs(tmp_path, [100, 200])
    encoder = (_FakeModel(), _FakeProcessor())

    emb = extract_embeddings("fake", paths, device="cpu", batch=2,
                             encoder=encoder, flips=("h", "v"), log_ram=False)

    # warna solid -> flip tak mengubah piksel -> rata-rata TTA = nilai asli
    assert emb.shape == (2, 4)
    assert np.allclose(emb[:, 0], [100, 200])


def test_extract_empty_input_returns_empty_array():
    encoder = (_FakeModel(), _FakeProcessor())
    emb = extract_embeddings("fake", [], device="cpu", encoder=encoder, log_ram=False)
    assert emb.shape[0] == 0

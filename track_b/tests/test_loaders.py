import numpy as np
import pandas as pd
import pytest
from PIL import Image

from config import make_cfg
from model import get_data_config
from loaders import get_loaders_b
from transforms import normalize_of


@pytest.fixture
def tiny_dataset(tmp_path):
    """5 fold x 4 gambar 32x32 -> cukup untuk menguji plumbing tanpa GPU."""
    rows = []
    for f in range(5):
        for i in range(4):
            p = tmp_path / f"f{f}_{i}.jpg"
            Image.new("RGB", (32, 32), color=(i * 40, 60, 90)).save(p)
            rows.append({"filepath": str(p), "label": i % 3, "fold": f})
    csv = tmp_path / "folds.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return str(csv)


def test_val_row_idx_matches_folds_csv_rows_for_that_fold(tiny_dataset):
    cfg = make_cfg(folds_csv=tiny_dataset, batch=2, num_workers=0, img_size=32)
    dc = get_data_config("convnext_tiny.in12k_ft_in1k")
    _, val_loader, val_idx = get_loaders_b(fold=0, cfg=cfg, data_config=dc)

    df = pd.read_csv(tiny_dataset)
    expected = df.index[df["fold"] == 0].to_numpy()
    assert np.array_equal(val_idx, expected)
    assert len(val_idx) == len(val_loader.dataset)


def test_train_loader_excludes_the_validation_fold(tiny_dataset):
    cfg = make_cfg(folds_csv=tiny_dataset, batch=2, num_workers=0, img_size=32)
    dc = get_data_config("convnext_tiny.in12k_ft_in1k")
    train_loader, val_loader, _ = get_loaders_b(fold=2, cfg=cfg, data_config=dc)
    assert len(train_loader.dataset) == 16   # 4 fold x 4
    assert len(val_loader.dataset) == 4


def test_loader_normalization_follows_the_backbone_not_imagenet(tiny_dataset):
    cfg = make_cfg(folds_csv=tiny_dataset, batch=2, num_workers=0, img_size=32)
    dc = get_data_config("tf_efficientnetv2_s.in21k_ft_in1k")
    train_loader, _, _ = get_loaders_b(fold=0, cfg=cfg, data_config=dc)
    mean, _ = normalize_of(train_loader.dataset.transform)
    assert np.allclose(mean, (0.5, 0.5, 0.5), atol=1e-3), \
        "loader masih memakai normalisasi ImageNet untuk EffNetV2"


def test_val_loader_is_not_shuffled(tiny_dataset):
    # Kalau val di-shuffle, val_row_idx tidak lagi align -> OOF diam-diam tertukar.
    from torch.utils.data import SequentialSampler
    cfg = make_cfg(folds_csv=tiny_dataset, batch=2, num_workers=0, img_size=32)
    dc = get_data_config("convnext_tiny.in12k_ft_in1k")
    _, val_loader, _ = get_loaders_b(fold=0, cfg=cfg, data_config=dc)
    assert isinstance(val_loader.sampler, SequentialSampler)

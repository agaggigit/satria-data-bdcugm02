import numpy as np
import pandas as pd
import pytest
from PIL import Image

from config import make_cfg
from transforms import normalize_of
from train import setup_run


@pytest.fixture
def tiny_dataset(tmp_path):
    rows = []
    for f in range(5):
        for i in range(4):
            p = tmp_path / f"f{f}_{i}.jpg"
            Image.new("RGB", (32, 32), color=(i * 40, 60, 90)).save(p)
            rows.append({"filepath": str(p), "label": i % 3, "fold": f})
    csv = tmp_path / "folds.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return str(csv)


def test_setup_run_builds_the_backbone_named_in_config(tiny_dataset):
    cfg = make_cfg(backbone="swin_tiny_patch4_window7_224.ms_in1k",
                    folds_csv=tiny_dataset, img_size=224, batch=2,
                    num_workers=0, pretrained=False)
    ctx = setup_run(cfg, fold=0)
    assert "swin" in type(ctx["model"]).__name__.lower(), \
        "cfg.backbone tidak tersambung ke build_model()"


def test_setup_run_wires_backbone_specific_normalization(tiny_dataset):
    cfg = make_cfg(backbone="tf_efficientnetv2_s.in21k_ft_in1k",
                    folds_csv=tiny_dataset, img_size=300, batch=2,
                    num_workers=0, pretrained=False)
    ctx = setup_run(cfg, fold=0)
    mean, _ = normalize_of(ctx["train_loader"].dataset.transform)
    assert np.allclose(mean, (0.5, 0.5, 0.5), atol=1e-3)
    assert np.allclose(ctx["data_config"]["mean"], (0.5, 0.5, 0.5), atol=1e-3)


def test_setup_run_returns_val_row_idx_for_oof(tiny_dataset):
    cfg = make_cfg(backbone="convnext_tiny.in12k_ft_in1k", folds_csv=tiny_dataset,
                    img_size=32, batch=2, num_workers=0, pretrained=False)
    ctx = setup_run(cfg, fold=3)
    df = pd.read_csv(tiny_dataset)
    assert np.array_equal(ctx["val_row_idx"], df.index[df["fold"] == 3].to_numpy())

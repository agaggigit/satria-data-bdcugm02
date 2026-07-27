"""test_senior_spec_compare.py — Unit test untuk senior_spec_compare.py"""
import numpy as np
import pandas as pd
import pytest

from senior_spec_compare import run_senior_spec_eval


def test_run_senior_spec_eval_returns_expected_keys():
    n_v1 = 100
    n_v2 = 90
    rng = np.random.RandomState(42)

    X_v1 = rng.randn(n_v1, 1536).astype(np.float32)
    X_v2 = X_v1[:n_v2]

    folds_v1 = pd.DataFrame({
        "filename": [f"img_{i}.jpg" for i in range(n_v1)],
        "label": rng.randint(0, 3, size=n_v1),
        "fold": [i % 5 for i in range(n_v1)]
    })
    folds_v2 = folds_v1.iloc[:n_v2].copy()

    res = run_senior_spec_eval("siglip2so400m", X_v1, folds_v1, X_v2, folds_v2, head_name="mlp", seed=42)

    assert res["combo"] == "siglip2so400m"
    assert res["head"] == "mlp"
    assert "mean_baseline_v1" in res
    assert "mean_naif_v2" in res
    assert "mean_adil_v2" in res
    assert "delta_adil_minus_baseline" in res
    assert 0.0 <= res["mean_baseline_v1"] <= 1.0

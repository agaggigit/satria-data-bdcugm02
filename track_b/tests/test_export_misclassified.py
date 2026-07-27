"""test_export_misclassified.py — Unit test untuk export_misclassified.py"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from export_misclassified import extract_misclassified_df


def test_extract_misclassified_df_returns_expected_structure():
    n_samples = 60
    rng = np.random.RandomState(42)

    X = rng.randn(n_samples, 64).astype(np.float32)
    folds_df = pd.DataFrame({
        "filename": [f"img_{i}.jpg" for i in range(n_samples)],
        "label": rng.randint(0, 3, size=n_samples),
        "fold": [i % 5 for i in range(n_samples)]
    })

    wrong_df, summary_df = extract_misclassified_df(X, folds_df, head_name="knn", seed=42)

    assert isinstance(wrong_df, pd.DataFrame)
    assert isinstance(summary_df, pd.DataFrame)
    assert "filename" in wrong_df.columns
    assert "true_label_name" in wrong_df.columns
    assert "pred_label_name" in wrong_df.columns
    assert "confidence" in wrong_df.columns
    assert (wrong_df["pred_label_id"] != wrong_df["label"]).all()

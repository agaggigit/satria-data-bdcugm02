import numpy as np
import pandas as pd
import pytest

from probe_cv import run_probe_cv

rng = np.random.default_rng(42)


@pytest.fixture
def data():
    n, d = 150, 10
    y = np.tile([0, 1, 2], n // 3)
    centers = rng.normal(size=(3, d)) * 5
    X = centers[y] + rng.normal(scale=0.3, size=(n, d))
    folds_df = pd.DataFrame({
        "filepath": [f"img{i}.jpg" for i in range(n)],
        "label": y,
        "fold": np.tile(np.arange(5), n // 5),
    })
    return X, folds_df


def test_oof_covers_every_row_exactly_once(data):
    X, folds_df = data
    oof, _ = run_probe_cv(X, folds_df, "linear")
    assert oof.shape == (len(folds_df), 3)
    assert not np.isnan(oof).any()


def test_oof_probabilities_sum_to_one(data):
    X, folds_df = data
    oof, _ = run_probe_cv(X, folds_df, "linear")
    assert np.allclose(oof.sum(axis=1), 1.0, atol=1e-5)


def test_returns_one_score_per_fold(data):
    X, folds_df = data
    _, scores = run_probe_cv(X, folds_df, "linear")
    assert len(scores) == 5


def test_separable_data_scores_high(data):
    X, folds_df = data
    oof, scores = run_probe_cv(X, folds_df, "linear")
    assert min(scores) > 0.9, "data jelas terpisah tapi CV rendah -- ada yang salah di plumbing"


def test_oof_row_i_predicted_by_a_model_that_never_saw_row_i(data):
    """Kalau OOF bocor (model melihat sampel validasinya), skornya jadi mustahil sempurna."""
    X, folds_df = data
    y = folds_df["label"].to_numpy()
    X_noise = rng.normal(size=X.shape)          # fitur acak, tidak ada sinyal
    oof, _ = run_probe_cv(X_noise, folds_df, "linear")
    acc = (oof.argmax(axis=1) == y).mean()
    assert acc < 0.6, "fitur acak tapi akurasi tinggi -- OOF bocor!"

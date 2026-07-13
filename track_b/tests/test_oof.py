import numpy as np
import pandas as pd
import pytest
from oof import assemble_oof, validate_oof


def make_folds_df(n_per_fold=4, n_folds=5):
    rows = []
    for f in range(n_folds):
        for i in range(n_per_fold):
            rows.append({"filepath": f"img_{f}_{i}.jpg", "label": i % 3, "fold": f})
    return pd.DataFrame(rows)


def test_assemble_oof_places_each_row_exactly_once():
    df = make_folds_df()
    n = len(df)
    fold_probs = {}
    for f in range(5):
        idx = df.index[df["fold"] == f].to_numpy()
        probs = np.full((len(idx), 3), 1 / 3)
        fold_probs[f] = (idx, probs)

    oof = assemble_oof(fold_probs, n_rows=n)
    assert oof.shape == (n, 3)
    assert not np.isnan(oof).any()  # tidak ada baris kosong


def test_assemble_oof_respects_row_index_not_order():
    # Fold dikembalikan dengan urutan acak -> hasil harus tetap align ke index asli.
    df = make_folds_df(n_per_fold=2, n_folds=2)
    n = len(df)
    idx0 = np.array([2, 0])          # sengaja tidak terurut
    probs0 = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    idx1 = np.array([1, 3])
    probs1 = np.array([[0.2, 0.2, 0.6], [0.3, 0.4, 0.3]])

    oof = assemble_oof({0: (idx0, probs0), 1: (idx1, probs1)}, n_rows=n)
    assert np.allclose(oof[2], [0.9, 0.05, 0.05])
    assert np.allclose(oof[0], [0.1, 0.8, 0.1])
    assert np.allclose(oof[1], [0.2, 0.2, 0.6])


def test_validate_oof_rejects_uncovered_rows():
    df = make_folds_df()
    oof = np.full((len(df), 3), 1 / 3)
    oof[7] = np.nan  # satu baris tidak pernah diisi
    with pytest.raises(AssertionError, match="belum terisi"):
        validate_oof(oof, df)


def test_validate_oof_rejects_probs_not_summing_to_one():
    df = make_folds_df()
    oof = np.full((len(df), 3), 0.5)  # sum = 1.5
    with pytest.raises(AssertionError, match="sum"):
        validate_oof(oof, df)


def test_validate_oof_accepts_valid_oof():
    df = make_folds_df()
    oof = np.full((len(df), 3), 1 / 3)
    validate_oof(oof, df)  # tidak raise

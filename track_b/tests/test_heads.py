import numpy as np
import pytest

from heads import HEAD_NAMES, make_head

rng = np.random.default_rng(42)


def _separable(n=90, d=8):
    y = np.repeat([0, 1, 2], n // 3)
    centers = rng.normal(size=(3, d)) * 5
    X = centers[y] + rng.normal(scale=0.3, size=(n, d))
    return X, y


@pytest.mark.parametrize("name", HEAD_NAMES)
def test_head_outputs_probabilities_shaped_n_by_3(name):
    X, y = _separable()
    head = make_head(name)
    head.fit(X, y)
    p = head.predict_proba(X)
    assert p.shape == (len(y), 3)


@pytest.mark.parametrize("name", HEAD_NAMES)
def test_head_probabilities_sum_to_one(name):
    X, y = _separable()
    head = make_head(name)
    head.fit(X, y)
    assert np.allclose(head.predict_proba(X).sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("name", HEAD_NAMES)
def test_head_learns_separable_data(name):
    X, y = _separable()
    head = make_head(name)
    head.fit(X, y)
    acc = (head.predict_proba(X).argmax(axis=1) == y).mean()
    assert acc > 0.9, f"{name} gagal di data yang jelas terpisah -- kemungkinan salah wiring"


def test_unknown_head_raises_immediately():
    with pytest.raises(KeyError, match="tidak dikenal"):
        make_head("transformer_sakti")

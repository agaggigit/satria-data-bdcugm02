import numpy as np

from features import concat_features, l2norm


def test_l2norm_rows_become_unit_length():
    X = np.array([[3.0, 4.0], [1.0, 0.0]])
    out = l2norm(X)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_concat_shape_is_sum_of_dims():
    a, b = np.random.rand(10, 4), np.random.rand(10, 6)
    assert concat_features([a, b]).shape == (10, 10)


def test_concat_normalizes_each_block_so_scale_cannot_dominate():
    # Blok kedua skalanya 1000x. Tanpa normalisasi per-blok, dia akan menelan blok pertama.
    a = np.random.rand(10, 4)
    b = np.random.rand(10, 4) * 1000.0
    out = concat_features([a, b])
    norm_a = np.linalg.norm(out[:, :4], axis=1)
    norm_b = np.linalg.norm(out[:, 4:], axis=1)
    assert np.allclose(norm_a, norm_b, atol=1e-6), "satu blok mendominasi -- normalisasi bocor"


def test_l2norm_handles_zero_rows_without_nan():
    X = np.zeros((3, 5))
    assert not np.isnan(l2norm(X)).any()

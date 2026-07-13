import numpy as np
from metrics import macro_f1, per_class_f1, confusion, margin, entropy


def test_macro_f1_perfect_prediction():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y, y) == 1.0


def test_macro_f1_counts_missing_class_as_zero_not_nan():
    # Kelas 1 (Electronic) tidak pernah diprediksi -> F1 kelas itu 0, bukan NaN.
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 0, 2])
    score = macro_f1(y_true, y_pred)
    assert not np.isnan(score)
    assert 0.0 < score < 1.0


def test_per_class_f1_returns_three_values_in_label_order():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 0, 1, 2])
    out = per_class_f1(y_true, y_pred)
    assert out.shape == (3,)
    assert np.allclose(out, [1.0, 1.0, 1.0])


def test_confusion_shape_is_always_3x3():
    y_true = np.array([0, 0])
    y_pred = np.array([0, 0])
    assert confusion(y_true, y_pred).shape == (3, 3)


def test_margin_is_top1_minus_top2():
    probs = np.array([[0.7, 0.2, 0.1]])
    assert np.allclose(margin(probs), [0.5])


def test_entropy_is_zero_for_confident_and_max_for_uniform():
    confident = np.array([[1.0, 0.0, 0.0]])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert np.allclose(entropy(confident), [0.0], atol=1e-6)
    assert np.allclose(entropy(uniform), [np.log(3)], atol=1e-6)

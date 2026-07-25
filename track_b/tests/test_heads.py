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


# --- B2 (TRACK_B_ARAHAN_V3.md): class_weight harus BENAR-BENAR berefek, bukan
# diterima lalu diabaikan diam-diam. Ini bug yang sebelumnya ada di MLP:
# make_head("mlp", class_weight=...) menerima parameter tapi MLPClassifier
# tidak punya class_weight di constructor -> nilainya lenyap tanpa error. ---

def _imbalanced(seed=42, d=6):
    """Rasio ~ mirip Electronic asli (kelas 1 minoritas ~15% dari total),
    dengan overlap cukup besar supaya head TANPA penyeimbang cenderung
    'menelan' kelas minoritas demi akurasi mayoritas."""
    rng = np.random.default_rng(seed)
    n0, n1, n2 = 200, 35, 200
    y = np.concatenate([np.zeros(n0), np.ones(n1), np.full(n2, 2)]).astype(int)
    centers = rng.normal(size=(3, d)) * 2.5
    X = centers[y] + rng.normal(scale=1.8, size=(len(y), d))
    return X, y


def _minority_recall(head, X, y, minority=1):
    pred = head.predict_proba(X).argmax(axis=1)
    mask = y == minority
    return (pred[mask] == minority).mean()


@pytest.mark.parametrize("name", ["linear", "mlp", "lgbm"])
def test_class_weight_changes_the_fitted_model(name):
    """Bobot minoritas yang BESAR harus mengubah prediksi -- kalau tidak,
    class_weight diam-diam diabaikan (persis bug yang baru diperbaiki di MLP)."""
    X, y = _imbalanced()

    head_plain = make_head(name, seed=42, class_weight=None)
    head_plain.fit(X, y)
    proba_plain = head_plain.predict_proba(X)

    head_weighted = make_head(name, seed=42, class_weight={0: 1, 1: 10, 2: 1})
    head_weighted.fit(X, y)
    proba_weighted = head_weighted.predict_proba(X)

    assert not np.allclose(proba_plain, proba_weighted, atol=1e-6), (
        f"{name}: class_weight={{1: 10}} tidak mengubah predict_proba sama sekali "
        f"-- kemungkinan class_weight diabaikan diam-diam"
    )


@pytest.mark.parametrize("name", ["linear", "mlp", "lgbm"])
def test_class_weight_balanced_does_not_hurt_minority_recall(name):
    X, y = _imbalanced()

    head_plain = make_head(name, seed=42, class_weight=None)
    head_plain.fit(X, y)
    recall_plain = _minority_recall(head_plain, X, y)

    head_balanced = make_head(name, seed=42, class_weight="balanced")
    head_balanced.fit(X, y)
    recall_balanced = _minority_recall(head_balanced, X, y)

    assert recall_balanced >= recall_plain, (
        f"{name}: class_weight='balanced' MENURUNKAN recall kelas minoritas "
        f"({recall_plain:.3f} -> {recall_balanced:.3f})"
    )


# --- Kompatibilitas lintas versi sklearn: MLPClassifier.fit(sample_weight)
# ADA di sklearn baru (lokal 1.9.0) tapi TIDAK di sklearn lama (Colab, 25 Juli
# 2026 -> TypeError). Memverifikasi signature cuma di satu mesin TIDAK CUKUP;
# test di bawah memaksa KEDUA jalur supaya bug ini tak bisa lolos lagi. ---

def test_mlp_detects_sample_weight_support_at_runtime():
    head = make_head("mlp", class_weight="balanced")
    assert isinstance(head.supports_sample_weight, bool)


def test_mlp_uses_sample_weight_path_when_supported():
    head = make_head("mlp", class_weight="balanced")
    if not head.supports_sample_weight:
        pytest.skip("sklearn di mesin ini tidak mendukung sample_weight")
    X, y = _imbalanced()
    head.fit(X, y)
    assert head.weighting_ == "sample_weight"


def test_mlp_falls_back_to_resampling_on_old_sklearn():
    """Simulasi sklearn lama: paksa supports_sample_weight=False. HARUS tetap
    jalan (tidak TypeError) DAN tetap menyeimbangkan kelas -- bukan diam-diam
    melatih tanpa bobot."""
    X, y = _imbalanced()
    head = make_head("mlp", seed=42, class_weight="balanced")
    head.supports_sample_weight = False          # <- seolah-olah sklearn Colab
    head.fit(X, y)

    assert head.weighting_ == "resample"
    p = head.predict_proba(X)
    assert p.shape == (len(y), 3)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_mlp_resample_fallback_still_balances_minority_class():
    """Fallback harus BENAR-BENAR menyeimbangkan: recall kelas minoritas tidak
    boleh lebih buruk daripada tanpa bobot sama sekali."""
    X, y = _imbalanced()

    plain = make_head("mlp", seed=42, class_weight=None)
    plain.fit(X, y)
    recall_plain = _minority_recall(plain, X, y)

    fallback = make_head("mlp", seed=42, class_weight="balanced")
    fallback.supports_sample_weight = False
    fallback.fit(X, y)
    recall_fallback = _minority_recall(fallback, X, y)

    assert fallback.weighting_ == "resample"
    assert recall_fallback >= recall_plain, (
        f"fallback resample MENURUNKAN recall minoritas "
        f"({recall_plain:.3f} -> {recall_fallback:.3f}) -- penyeimbang tidak bekerja"
    )


def test_mlp_resample_fallback_is_reproducible():
    """Seed sama -> hasil sama persis (resampling tidak boleh bikin run
    tak-reproducible; panitia bisa minta verifikasi)."""
    X, y = _imbalanced()

    def _run():
        h = make_head("mlp", seed=42, class_weight="balanced")
        h.supports_sample_weight = False
        h.fit(X, y)
        return h.predict_proba(X)

    assert np.allclose(_run(), _run())


def test_mlp_without_class_weight_records_none_path():
    X, y = _imbalanced()
    head = make_head("mlp", seed=42, class_weight=None)
    head.fit(X, y)
    assert head.weighting_ == "none"


def test_knn_does_not_accept_class_weight_and_make_head_does_not_pass_it():
    """KNeighborsClassifier TIDAK punya param class_weight -- kalau make_head
    mengirimkannya, ini akan TypeError. Sengaja mengirim class_weight ke knn
    dan pastikan tidak meledak (dibuktikan lewat inspect.signature juga)."""
    import inspect
    from sklearn.neighbors import KNeighborsClassifier
    assert "class_weight" not in inspect.signature(KNeighborsClassifier.__init__).parameters

    head = make_head("knn", class_weight="balanced")   # tidak boleh raise
    X, y = _imbalanced()
    head.fit(X, y)   # tidak boleh raise

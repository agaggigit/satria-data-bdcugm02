"""heads.py — Interface terpadu untuk head murah di atas embedding beku.

Satu interface (.fit / .predict_proba) untuk semua head -> grid backbone x head
jadi loop sederhana, dan menambah head baru tidak menyentuh kode lain.
"""
import inspect

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

HEAD_NAMES = ["linear", "mlp", "lgbm", "knn"]


class _MLPWrapper:
    """MLPClassifier tidak punya param `class_weight` di constructor -- kalau
    dikirim langsung, TypeError. Wrapper ini yang menerjemahkannya, supaya MLP
    -- head UTAMA (A1) -- tidak diam-diam kehilangan penyeimbang kelas
    Electronic (bug B2).

    DUA JALUR, dipilih saat runtime (bukan asumsi):
      sample_weight : sklearn baru (>=1.7-an) menerima fit(X, y, sample_weight)
      resample      : sklearn lama TIDAK menerimanya (Colab per 25 Juli 2026
                      masih begitu -> TypeError 'unexpected keyword argument
                      sample_weight'). Jalur ini menyeimbangkan lewat resampling
                      berbobot: ambil ulang n sampel dgn probabilitas
                      sebanding class_weight, seeded jadi reproducible.

    Sengaja TIDAK jatuh ke "latih tanpa bobot" kalau sample_weight tak ada --
    itu persis menghidupkan lagi bug yang baru diperbaiki, dan diam-diam pula.
    Jalur yang dipakai tercatat di `self.weighting_` untuk audit/report."""

    def __init__(self, seed: int, class_weight):
        self.class_weight = class_weight
        self.seed = seed
        self.model = MLPClassifier(hidden_layer_sizes=(512,), max_iter=400,
                                   early_stopping=True, random_state=seed)
        self.supports_sample_weight = "sample_weight" in inspect.signature(
            self.model.fit).parameters
        self.weighting_ = None

    def fit(self, X, y):
        if self.class_weight is None:
            self.weighting_ = "none"
            self.model.fit(X, y)
            return self

        sw = compute_sample_weight(self.class_weight, y)
        if self.supports_sample_weight:
            try:
                self.model.fit(X, y, sample_weight=sw)
                self.weighting_ = "sample_weight"
                return self
            except TypeError:
                # Jaring pengaman kedua: introspeksi signature bilang didukung
                # tapi implementasinya menolak (dekorator sklearn menyembunyikan
                # signature asli, versi tak lazim, dsb). Jangan gagal total --
                # turun ke jalur resample yang sama-sama menyeimbangkan.
                self.supports_sample_weight = False

        self.weighting_ = "resample"
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(y), size=len(y), replace=True, p=sw / sw.sum())
        self.model.fit(np.asarray(X)[idx], np.asarray(y)[idx])
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)


class _LGBMWrapper:
    """Bungkus LightGBM supaya interface-nya sama dengan head lain."""

    def __init__(self, seed: int, class_weight):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=300,
            learning_rate=0.05, num_leaves=31, random_state=seed,
            class_weight=class_weight, verbose=-1,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)


def make_head(name: str, seed: int = 42, class_weight=None):
    if name == "linear":
        return LogisticRegression(max_iter=3000, C=1.0,
                                  class_weight=class_weight, random_state=seed)
    if name == "mlp":
        return _MLPWrapper(seed, class_weight)
    if name == "lgbm":
        return _LGBMWrapper(seed, class_weight)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=15, metric="cosine", weights="distance")
    raise KeyError(f"head '{name}' tidak dikenal. Pilihan: {HEAD_NAMES}")

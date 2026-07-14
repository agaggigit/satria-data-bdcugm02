"""heads.py — Interface terpadu untuk head murah di atas embedding beku.

Satu interface (.fit / .predict_proba) untuk semua head -> grid backbone x head
jadi loop sederhana, dan menambah head baru tidak menyentuh kode lain.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

HEAD_NAMES = ["linear", "mlp", "lgbm", "knn"]


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
        return MLPClassifier(hidden_layer_sizes=(512,), max_iter=400,
                             early_stopping=True, random_state=seed)
    if name == "lgbm":
        return _LGBMWrapper(seed, class_weight)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=15, metric="cosine", weights="distance")
    raise KeyError(f"head '{name}' tidak dikenal. Pilihan: {HEAD_NAMES}")

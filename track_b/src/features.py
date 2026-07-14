"""features.py — L2-normalize embedding blocks lalu gabung.

Kalau normalisasi dilakukan SETELAH concat (atau tidak sama sekali), blok
dengan norm lebih besar mendominasi dan blok lain jadi hiasan -- "menggabungkan"
tiga backbone tapi efektifnya cuma memakai satu.
"""
import numpy as np


def l2norm(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)     # jangan bagi nol
    return X / norms


def concat_features(blocks: list) -> np.ndarray:
    """L2-norm TIAP blok dulu, baru gabung. Urutan ini tidak boleh dibalik."""
    n = blocks[0].shape[0]
    for b in blocks:
        assert b.shape[0] == n, "blok embedding tidak sepanjang yang lain -- alignment rusak"
    return np.hstack([l2norm(b) for b in blocks])

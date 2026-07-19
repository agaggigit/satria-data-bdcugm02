"""ensemble_embedding.py — Ensemble inference berbasis embedding cache.

PERUBAHAN ARSITEKTUR vs ensemble_tta.py (v1):
- v1: load 5x checkpoint .pt ConvNeXt → forward image test → rata-rata prob
- v2: load embedding test .npy → re-fit head di seluruh train → predict_proba

Keuntungan v2:
- Tidak perlu GPU untuk inference test (head = KNN/linear, berjalan di CPU menit)
- Tidak ada risiko mismatch transform test vs train (embedding sudah di-cache)
- Bisa menggabungkan BANYAK backbone dalam hitungan detik
- Tidak bergantung pada timm atau checkpoint .pt apapun

Pipeline untuk tiap komposisi:
    1. Load emb_train = embedding train dari cache (align folds_v2.csv)
    2. Load emb_test  = embedding test dari cache (1458 gambar, align submission.csv)
    3. Fit head (KNN/linear/etc) di SELURUH train bersih
    4. predict_proba(emb_test) → prob_test [1458, 3]
    5. Gabungkan antar komposisi dengan bobot → prob_ensemble [1458, 3]
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path


def _load_emb(path: str, split_name: str) -> np.ndarray:
    """Load embedding .npy dengan validasi dasar."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Embedding {split_name} tidak ditemukan: {path}\n"
            f"Pastikan Track B sudah mengekstrak dan menyimpan embedding ke Drive."
        )
    emb = np.load(path)
    assert emb.ndim == 2, f"Embedding {split_name} harus 2D, dapat {emb.ndim}D"
    assert np.isfinite(emb).all(), f"Embedding {split_name} mengandung NaN/inf"
    return emb.astype(np.float32)


def _get_train_embeddings(comp_cfg, folds_df: pd.DataFrame) -> np.ndarray:
    """Load embedding train dari cache Drive.

    Track B menyimpan embedding train di EMB_DIR dengan format:
        {backbone_shortname}_train.npy
    Contoh: siglip2so400m_train.npy

    Fallback: kalau file train tidak ada, coba reconstruct dari OOF index.
    Tapi sebaiknya Track B menyediakan file ini secara eksplisit.
    """
    from track_c.src.config_c import EMB_DIR

    # Derive nama backbone dari oof path
    oof_basename = os.path.basename(comp_cfg.oof)
    # Contoh: oof_siglip2so400m_knn_v2.npy → siglip2so400m
    # Ekstrak bagian antara oof_ dan _{head}
    name_part = oof_basename.replace("oof_", "").split("_knn")[0].split("_linear")[0].split("_mlp")[0].split("_lgbm")[0]

    # Path kandidat embedding train
    train_emb_path = os.path.join(EMB_DIR, f"{name_part}_train.npy")
    if not os.path.exists(train_emb_path):
        # Coba alternatif langsung di OUTPUT_TRACK_B
        from track_c.src.config_c import OUTPUT_TRACK_B
        train_emb_path = os.path.join(OUTPUT_TRACK_B, f"{name_part}_train.npy")

    emb_train = _load_emb(train_emb_path, "train")
    assert emb_train.shape[0] == len(folds_df), (
        f"Jumlah baris embedding train ({emb_train.shape[0]}) "
        f"!= jumlah baris folds_v2.csv ({len(folds_df)}). "
        f"Pastikan embedding align dengan folds_v2.csv!"
    )
    return emb_train


def apply_composition(comp_cfg, folds_df: pd.DataFrame,
                       composition_weights: np.ndarray = None) -> np.ndarray:
    """Fit head di seluruh train bersih → predict_proba di test.

    Args:
        comp_cfg  : SimpleNamespace dari COMPOSITIONS[key]
        folds_df  : DataFrame folds_v2.csv (label, fold, filepath)
        composition_weights: tidak dipakai di sini (untuk multi-komposisi)

    Returns:
        prob_test : [1458, 3] probabilitas untuk test data
    """
    # 1. Load embedding test
    emb_test = _load_emb(comp_cfg.emb_test, "test")
    print(f"  emb_test shape: {emb_test.shape}")

    # 2. Load embedding train
    emb_train = _get_train_embeddings(comp_cfg, folds_df)
    print(f"  emb_train shape: {emb_train.shape}")

    # 3. Fit head di SELURUH train bersih
    y_train = folds_df["label"].values
    assert set(np.unique(y_train)) <= {0, 1, 2}, \
        f"Label train di luar {{0,1,2}}: {np.unique(y_train)}"

    # Import head dari Track B (re-use, jangan tulis ulang)
    _ensure_track_b_in_path()
    from heads import make_head
    head = make_head(comp_cfg.head, seed=42, class_weight=None)

    print(f"  Fitting {comp_cfg.head} head di {len(y_train)} sampel...")
    head.fit(emb_train, y_train)

    # 4. Predict probabilitas test
    prob_test = head.predict_proba(emb_test)
    assert prob_test.shape == (emb_test.shape[0], 3), \
        f"Shape prob_test salah: {prob_test.shape}"

    return prob_test.astype(np.float32)


def run_ensemble_inference(cfg, folds_df: pd.DataFrame,
                            ensemble_weights: dict = None) -> np.ndarray:
    """Jalankan semua komposisi aktif → rata-ratakan dengan bobot.

    Args:
        cfg           : CFG_C
        folds_df      : DataFrame folds_v2.csv
        ensemble_weights: dict {comp_key: float} atau None (rata-rata seragam)

    Returns:
        prob_ensemble : [1458, 3]
    """
    active = cfg.active_compositions
    n_comp = len(active)
    print(f"\nMemulai ensemble inference: {n_comp} komposisi aktif")
    print(f"  Komposisi: {active}\n")

    # Default: bobot seragam
    if ensemble_weights is None:
        weights = {k: 1.0 / n_comp for k in active}
    else:
        total = sum(ensemble_weights.values())
        weights = {k: v / total for k, v in ensemble_weights.items()}

    prob_ensemble = None
    for key in active:
        comp_cfg = cfg.compositions[key]
        w = weights.get(key, 1.0 / n_comp)
        print(f"\n--- Komposisi: {key} (bobot={w:.3f}) ---")
        print(f"    Backbone: {comp_cfg.backbone}")
        print(f"    Head    : {comp_cfg.head}")

        prob_test = apply_composition(comp_cfg, folds_df)

        if prob_ensemble is None:
            prob_ensemble = np.zeros_like(prob_test)
        prob_ensemble += w * prob_test

    assert prob_ensemble is not None, "Tidak ada komposisi aktif!"
    print(f"\nEnsemble selesai. Shape: {prob_ensemble.shape}")
    return prob_ensemble


# ---------------------------------------------------------------------------
# Helper: pastikan track_b/src ada di sys.path (untuk import heads.py)
# ---------------------------------------------------------------------------

def _ensure_track_b_in_path():
    """Tambahkan track_b/src ke sys.path agar bisa import heads.py dari Track B.

    heads.py tidak di-copy ke track_c agar tidak ada dua sumber kebenaran.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    track_b_src = os.path.join(repo_root, "track_b", "src")
    if track_b_src not in sys.path:
        sys.path.insert(0, track_b_src)


# ---------------------------------------------------------------------------
# Validasi embedding test alignment terhadap submission template
# ---------------------------------------------------------------------------

def assert_test_alignment(emb_test: np.ndarray, template_df: pd.DataFrame):
    """Pastikan jumlah baris embedding test = jumlah baris submission template."""
    n_test = len(template_df)
    assert emb_test.shape[0] == n_test, (
        f"Jumlah baris embedding test ({emb_test.shape[0]}) "
        f"!= submission template ({n_test}). "
        f"Urutan gambar test WAJIB sama persis dengan submission.csv panitia!"
    )
    print(f"  ✅ Alignment OK: {n_test} baris test embedding = {n_test} submission rows")


if __name__ == "__main__":
    # Smoke test tanpa GPU (hanya shape check)
    print("Smoke test ensemble_embedding.py ...")
    # Buat dummy data
    np.random.seed(42)
    N_train, N_test, D = 200, 20, 64
    dummy_emb_train = np.random.randn(N_train, D).astype(np.float32)
    dummy_emb_test  = np.random.randn(N_test, D).astype(np.float32)
    dummy_labels    = np.random.choice([0, 1, 2], N_train)

    # Simpan ke tmp
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "train_emb.npy"), dummy_emb_train)
        np.save(os.path.join(tmpdir, "test_emb.npy"),  dummy_emb_test)

        # Test head langsung
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "track_b", "src"))
        try:
            from heads import make_head
            head = make_head("knn", seed=42)
            head.fit(dummy_emb_train, dummy_labels)
            probs = head.predict_proba(dummy_emb_test)
            assert probs.shape == (N_test, 3), f"Shape salah: {probs.shape}"
            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "Prob tidak sum ke 1"
            print(f"  KNN head OK: {probs.shape}")

            head_lin = make_head("linear", seed=42)
            head_lin.fit(dummy_emb_train, dummy_labels)
            probs_lin = head_lin.predict_proba(dummy_emb_test)
            assert probs_lin.shape == (N_test, 3)
            print(f"  Linear head OK: {probs_lin.shape}")

        except ImportError:
            print("  [SKIP] heads.py dari track_b/src tidak tersedia di lingkungan ini")

    print("Smoke test PASSED")

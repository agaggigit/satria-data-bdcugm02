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
    1. Load emb_train = embedding train dari cache (align folds.csv ASLI, 26527 baris)
    2. Filter emb_train ke baris yang ada di folds_v2.csv (25985 baris bersih)
    3. Load emb_test  = embedding test dari cache (1458 gambar, align submission.csv)
    4. Fit head (KNN/linear/etc) di data bersih saja
    5. predict_proba(emb_test) → prob_test [1458, 3]
    6. Gabungkan antar komposisi dengan bobot → prob_ensemble [1458, 3]

CATATAN PENTING — Kenapa ada langkah filter:
    Track B mengekstrak embedding dari SEMUA 26527 gambar train (align folds.csv asli).
    Setelah Track A cleaning, folds_v2.csv hanya punya 25985 baris (542 dihapus).
    OOF v2 sudah align 25985, tapi embedding cache masih 26527.
    Solusinya: filter embedding via filepath matching antara folds.csv (26527)
    dan folds_v2.csv (25985) — tanpa perlu Track B regenerate embedding.
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


def _find_train_emb_path(comp_cfg) -> str:
    """Cari path file embedding train dari beberapa lokasi kandidat."""
    from track_c.src.config_c import EMB_DIR, OUTPUT_TRACK_B

    # Derive nama backbone dari nama file OOF
    # Contoh: oof_siglip2so400m_knn_v2.npy → siglip2so400m
    oof_basename = os.path.basename(comp_cfg.oof)
    name_part = (oof_basename
                 .replace("oof_", "")
                 .split("_knn")[0]
                 .split("_linear")[0]
                 .split("_mlp")[0]
                 .split("_lgbm")[0])

    # Coba beberapa lokasi kandidat
    candidates = [
        os.path.join(EMB_DIR, f"{name_part}_train.npy"),
        os.path.join(OUTPUT_TRACK_B, f"{name_part}_train.npy"),
        os.path.join(EMB_DIR, f"{name_part}_folds.npy"),   # nama alternatif
        os.path.join(EMB_DIR, f"{name_part}_all.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Embedding train tidak ditemukan. Dicoba di:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\n\nSolusi:\n"
        + "  1. Tanyakan ke Track B nama persis file embedding train."
        + "  2. Override path di Colab: comp_cfg.emb_train = '/path/ke/file.npy'"
        + "  3. Atau set CFG_C.compositions['siglip2_knn'].emb_train = '...'"
    )


def _filter_embeddings_by_filepath(
    emb_full: np.ndarray,
    folds_original_df: pd.DataFrame,
    folds_clean_df: pd.DataFrame,
) -> np.ndarray:
    """Filter embedding 26527 → 25985 berdasarkan filepath matching.

    Strategi:
    - folds.csv asli (26527): baris ke-i = embedding ke-i
    - folds_v2.csv bersih (25985): subset dari folds.csv asli, beberapa baris dihapus
    - Cocokkan filepath antara keduanya untuk dapat index baris di emb_full

    Kolom yang dicocokkan: 'filepath' (atau 'path' / 'filename' — coba keduanya).
    """
    # Cari nama kolom filepath yang tersedia
    fp_col = None
    for col in ["filepath", "path", "filename", "file", "img_path"]:
        if col in folds_original_df.columns and col in folds_clean_df.columns:
            fp_col = col
            break

    if fp_col is None:
        raise KeyError(
            f"Tidak bisa menemukan kolom filepath yang sama di folds.csv dan folds_v2.csv.\n"
            f"Kolom folds.csv asli  : {list(folds_original_df.columns)}\n"
            f"Kolom folds_v2.csv    : {list(folds_clean_df.columns)}\n"
            f"Solusi: pastikan kedua CSV punya kolom yang sama untuk identifikasi gambar."
        )

    # Buat mapping filepath → index baris di folds_original_df
    # Normalisasi path: pakai basename saja untuk menghindari perbedaan prefix
    orig_paths = folds_original_df[fp_col].apply(
        lambda p: os.path.normpath(str(p))
    ).tolist()
    path_to_idx = {p: i for i, p in enumerate(orig_paths)}

    # Cari index di folds_original untuk setiap baris di folds_v2
    clean_paths = folds_clean_df[fp_col].apply(
        lambda p: os.path.normpath(str(p))
    ).tolist()

    indices = []
    not_found = []
    for p in clean_paths:
        if p in path_to_idx:
            indices.append(path_to_idx[p])
        else:
            not_found.append(p)

    if not_found:
        raise ValueError(
            f"{len(not_found)} filepath di folds_v2.csv tidak ditemukan di folds.csv asli.\n"
            f"Contoh: {not_found[:3]}\n"
            f"Ini tidak seharusnya terjadi — periksa apakah folds_v2.csv benar-benar "
            f"berasal dari folds.csv yang sama."
        )

    assert len(indices) == len(folds_clean_df), (
        f"Jumlah index cocok ({len(indices)}) != baris folds_v2.csv ({len(folds_clean_df)})"
    )

    return emb_full[np.array(indices)]


def _get_train_embeddings(comp_cfg, folds_df: pd.DataFrame,
                          folds_original_csv: str = None) -> np.ndarray:
    """Load dan (bila perlu) filter embedding train ke data bersih.

    Menangani dua kasus:
    A) emb_train sudah align folds_v2.csv (25985 baris) → langsung pakai
    B) emb_train align folds.csv asli (26527 baris) → filter via filepath matching

    Args:
        comp_cfg          : SimpleNamespace dari COMPOSITIONS[key]
        folds_df          : DataFrame folds_v2.csv yang sudah di-load (25985 baris)
        folds_original_csv: path ke folds.csv ASLI (26527 baris).
                            Diperlukan hanya kalau emb_train punya 26527 baris.
                            Kalau None, dicoba dari CFG_C.folds_csv_original.
    """
    from track_c.src.config_c import CFG_C, OUTPUT_TRACK_A

    # Cek apakah komposisi punya field emb_train eksplisit (override dari user)
    train_emb_path = getattr(comp_cfg, "emb_train", None)
    if train_emb_path is None:
        train_emb_path = _find_train_emb_path(comp_cfg)

    emb_full = _load_emb(train_emb_path, "train")
    n_clean = len(folds_df)
    n_full  = emb_full.shape[0]

    print(f"  emb_train shape  : {emb_full.shape}")
    print(f"  folds_v2 baris   : {n_clean}")

    # --- Kasus A: ukuran sudah sama ---
    if n_full == n_clean:
        print(f"  ✅ Embedding train sudah align folds_v2.csv ({n_clean} baris)")
        return emb_full

    # --- Kasus B: embedding 26527, folds_v2 25985 → filter ---
    print(f"  ⚠️  Ukuran tidak sama ({n_full} vs {n_clean}).")
    print(f"     Embedding di-ekstrak dari dataset sebelum cleaning.")
    print(f"     Memfilter {n_full} → {n_clean} baris via filepath matching...")

    # Cari folds.csv asli (26527)
    if folds_original_csv is None:
        folds_original_csv = getattr(CFG_C, "folds_csv_original",
                                     os.path.join(OUTPUT_TRACK_A, "folds.csv"))

    if not os.path.exists(folds_original_csv):
        raise FileNotFoundError(
            f"folds.csv ASLI (26527 baris) tidak ditemukan: {folds_original_csv}\n"
            f"File ini diperlukan untuk filter embedding train.\n"
            f"Solusi di Colab:\n"
            f"  CFG_C.folds_csv_original = '/content/drive/MyDrive/BDC2026apace/output_trackA/folds.csv'"
        )

    folds_original_df = pd.read_csv(folds_original_csv)
    assert len(folds_original_df) == n_full, (
        f"folds.csv asli ({len(folds_original_df)} baris) != embedding ({n_full} baris). "
        f"Pastikan folds.csv yang dipakai adalah versi yang sama dengan saat embedding di-ekstrak."
    )

    emb_filtered = _filter_embeddings_by_filepath(emb_full, folds_original_df, folds_df)
    assert emb_filtered.shape[0] == n_clean, (
        f"Setelah filter: {emb_filtered.shape[0]} baris (harusnya {n_clean})"
    )
    print(f"  ✅ Filter berhasil: {n_full} → {emb_filtered.shape[0]} baris")
    return emb_filtered


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

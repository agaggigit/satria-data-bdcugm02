"""
utils.py — Track A Helper Functions
BDC Satria Data 2026
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm  # fallback untuk environment non-notebook

MAX_WORKERS = 16  # default thread count untuk semua operasi paralel


# ─── Label Mapping ────────────────────────────────────────────────────────────

LABEL_MAP = {
    "0_Recyclable": 0,
    "1_Electronic": 1,
    "2_Organic": 2,
}
IDX_TO_CLASS = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = ["Recyclable", "Electronic", "Organic"]


# ─── Dataset Scanning ─────────────────────────────────────────────────────────

def scan_train_dir(train_dir: str) -> pd.DataFrame:
    """
    Scan direktori train dan return DataFrame dengan kolom:
    filepath (str), label_name (str), label (int)
    """
    train_dir = Path(train_dir)
    records = []
    for folder_name, label_idx in LABEL_MAP.items():
        class_dir = train_dir / folder_name
        assert class_dir.exists(), f"Folder tidak ditemukan: {class_dir}"
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                records.append({
                    "filepath": str(img_path),
                    "label_name": folder_name,
                    "label": label_idx,
                })
    df = pd.DataFrame(records)
    print(f"[scan_train_dir] Total gambar ditemukan: {len(df)}")
    return df


def scan_test_dir(test_dir: str) -> pd.DataFrame:
    """
    Scan direktori test dan return DataFrame terurut sesuai submission.csv.
    Kolom: filepath (str)
    """
    test_dir = Path(test_dir)
    records = []
    for img_path in sorted(test_dir.iterdir()):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            records.append({"filepath": str(img_path), "filename": img_path.name})
    df = pd.DataFrame(records)
    print(f"[scan_test_dir] Total gambar test: {len(df)}")
    return df


# ─── Image Health Checks ──────────────────────────────────────────────────────

def check_corrupt(
    df: pd.DataFrame,
    verbose: bool = True,
    max_workers: int = MAX_WORKERS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cek gambar yang tidak bisa dibuka secara paralel.
    Menggunakan img.load() — lebih relevan dari verify() karena
    ini yang dilakukan DataLoader saat training.
    Return (df_clean, skip_list).
    """
    def _check_one(args: Tuple[int, str]) -> Tuple[int, Optional[str]]:
        idx, fp = args
        try:
            with Image.open(fp) as img:
                img.load()
            return idx, None
        except Exception:
            return idx, fp

    filepaths = df["filepath"].tolist()
    skip_list: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_check_one, (idx, fp)): idx
                   for idx, fp in enumerate(filepaths)}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Cek corrupt", disable=not verbose):
            _, result = future.result()
            if result is not None:
                skip_list.append(result)

    if verbose:
        print(f"[check_corrupt] Corrupt / tidak terbaca: {len(skip_list)} file")
    df_clean = df[~df["filepath"].isin(set(skip_list))].reset_index(drop=True)
    return df_clean, skip_list


def get_image_stats(
    df: pd.DataFrame,
    sample_n: Optional[int] = None,
    max_workers: int = MAX_WORKERS,
) -> pd.DataFrame:
    """
    Kumpulkan statistik dimensi gambar secara paralel.
    Hanya membaca header file (tidak load pixel) — sangat cepat.
    Bisa di-sample untuk efisiensi lebih lanjut.
    Return DataFrame dengan kolom: filepath, width, height, channels, aspect_ratio
    """
    def _stats_one(args: Tuple[int, str]):
        idx, fp = args
        try:
            with Image.open(fp) as img:
                w, h = img.size          # hanya baca header
                c = len(img.getbands())
            return idx, fp, w, h, c, round(w / h, 3)
        except Exception:
            return idx, fp, None, None, None, None

    sample = df if sample_n is None else df.sample(min(sample_n, len(df)), random_state=42)
    filepaths = sample["filepath"].tolist()
    results = [None] * len(filepaths)  # pre-allocate untuk jaga urutan

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_stats_one, (idx, fp)): idx
                   for idx, fp in enumerate(filepaths)}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Statistik gambar"):
            idx, fp, w, h, c, ar = future.result()
            results[idx] = (fp, w, h, c, ar)

    rows = [
        {"filepath": fp, "width": w, "height": h, "channels": c, "aspect_ratio": ar}
        for fp, w, h, c, ar in results
        if w is not None
    ]
    return pd.DataFrame(rows)


# ─── Duplicate Detection ──────────────────────────────────────────────────────

def compute_phash(filepath: str, hash_size: int = 8) -> Optional[str]:
    """Hitung perceptual hash (pHash) sebuah gambar."""
    try:
        import imagehash
        with Image.open(filepath) as img:
            return str(imagehash.phash(img, hash_size=hash_size))
    except Exception:
        return None


def find_duplicates(
    df: pd.DataFrame,
    hash_size: int = 8,
    threshold: int = 0,
    max_workers: int = MAX_WORKERS,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Temukan grup duplikat/near-duplikat menggunakan perceptual hash secara paralel.

    Args:
        df        : DataFrame dengan kolom 'filepath'
        hash_size : ukuran pHash (default 8, jangan diubah sembarangan)
        threshold : max hamming distance yang dianggap duplikat
                    0 = hanya identik persis (cepat, O(n))
                    >0 = near-duplicate (lambat, O(n²)) — hindari untuk dataset besar
        max_workers: jumlah thread paralel

    Returns:
        (df_with_phash, dup_groups)
        df_with_phash : df asli + kolom 'phash'
        dup_groups    : dict {hash_str: [filepath, ...]} hanya yang duplikat
    """
    import imagehash

    def _phash_one(args: Tuple[int, str]):
        idx, fp = args
        try:
            with Image.open(fp) as img:
                return idx, str(imagehash.phash(img, hash_size=hash_size))
        except Exception:
            return idx, None

    filepaths = df["filepath"].tolist()
    hashes: List[Optional[str]] = [None] * len(filepaths)  # pre-allocate, jaga urutan

    print("[find_duplicates] Menghitung pHash secara paralel...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_phash_one, (idx, fp)): idx
                   for idx, fp in enumerate(filepaths)}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Menghitung pHash"):
            idx, h = future.result()
            hashes[idx] = h

    df = df.copy()
    df["phash"] = hashes

    # Deteksi duplikat
    dup_groups: Dict[str, List[str]] = {}
    if threshold == 0:
        # Exact match — cepat via value_counts
        hash_counts = df["phash"].value_counts()
        for h, cnt in hash_counts[hash_counts > 1].items():
            dup_groups[h] = df[df["phash"] == h]["filepath"].tolist()
    else:
        # Near-duplicate via hamming distance — O(n²), hanya untuk dataset kecil
        valid = df.dropna(subset=["phash"]).copy()
        hash_objs = [imagehash.hex_to_hash(h) for h in valid["phash"]]
        visited: set = set()
        for i, (fp_i, h_i) in enumerate(zip(valid["filepath"], hash_objs)):
            if i in visited:
                continue
            group = [fp_i]
            for j, (fp_j, h_j) in enumerate(zip(valid["filepath"], hash_objs)):
                if i != j and j not in visited and (h_i - h_j) <= threshold:
                    group.append(fp_j)
                    visited.add(j)
            if len(group) > 1:
                dup_groups[fp_i] = group
            visited.add(i)

    print(f"[find_duplicates] Grup duplikat ditemukan: {len(dup_groups)}")
    return df, dup_groups


def assign_group_ids(df: pd.DataFrame, dup_groups: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Tambahkan kolom 'group_id' ke df.
    - Gambar dalam grup duplikat → group_id yang sama (0, 1, 2, ...)
    - Gambar unik → group_id unik yang tidak bentrok dengan grup duplikat
    Dipakai oleh StratifiedGroupKFold agar kembaran tidak terpisah antar fold.
    """
    # Bangun mapping hash → group_id (vektorized, jauh lebih cepat dari apply per-baris)
    hash_to_group = {h: gid for gid, h in enumerate(dup_groups.keys())}
    n_dup_groups  = len(hash_to_group)

    df = df.copy()
    # Gambar dalam grup duplikat mendapat group_id dari hash-nya
    df["group_id"] = df["phash"].map(hash_to_group)

    # Gambar unik (NaN) mendapat ID unik mulai setelah semua grup duplikat
    mask_no_dup = df["group_id"].isna()
    df.loc[mask_no_dup, "group_id"] = range(n_dup_groups, n_dup_groups + mask_no_dup.sum())
    df["group_id"] = df["group_id"].astype(int)
    return df


# ─── Class Weights ────────────────────────────────────────────────────────────

def compute_class_weights(labels: List[int], n_classes: int = 3) -> np.ndarray:
    """
    Hitung class weights = n_samples / (n_classes * count_per_class).
    Siap dipakai di CrossEntropyLoss(weight=...).
    """
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    weights = len(labels) / (n_classes * counts)
    print(f"[compute_class_weights] {dict(zip(CLASS_NAMES, weights.round(4)))}")
    return weights


# ─── I/O Helpers ──────────────────────────────────────────────────────────────

def save_eda_stats(stats: dict, path: str = "outputs/eda_stats.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[save_eda_stats] Disimpan ke {path}")


def save_class_weights(weights: np.ndarray, path: str = "outputs/class_weights.npy"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, weights)
    print(f"[save_class_weights] Disimpan ke {path}")

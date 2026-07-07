"""
utils.py — Track A Helper Functions
BDC Satria Data 2026
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


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

def check_corrupt(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cek gambar yang tidak bisa dibuka. Return (df_clean, skip_list).
    """
    skip_list = []
    for fp in df["filepath"]:
        try:
            with Image.open(fp) as img:
                img.verify()
        except (UnidentifiedImageError, Exception):
            skip_list.append(fp)
    if verbose:
        print(f"[check_corrupt] Corrupt / tidak terbaca: {len(skip_list)} file")
    df_clean = df[~df["filepath"].isin(skip_list)].reset_index(drop=True)
    return df_clean, skip_list


def get_image_stats(df: pd.DataFrame, sample_n: Optional[int] = None) -> pd.DataFrame:
    """
    Kumpulkan statistik dimensi gambar. Bisa di-sample untuk efisiensi.
    Return DataFrame dengan kolom: filepath, width, height, channels, aspect_ratio
    """
    sample = df if sample_n is None else df.sample(min(sample_n, len(df)), random_state=42)
    rows = []
    for fp in sample["filepath"]:
        try:
            with Image.open(fp) as img:
                w, h = img.size
                c = len(img.getbands())
                rows.append({"filepath": fp, "width": w, "height": h,
                              "channels": c, "aspect_ratio": round(w / h, 3)})
        except Exception:
            pass
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


def find_duplicates(df: pd.DataFrame, hash_size: int = 8,
                    threshold: int = 5) -> Dict[str, List[str]]:
    """
    Temukan grup duplikat/near-duplikat menggunakan perceptual hash.
    threshold: max hamming distance yang dianggap duplikat (0 = identik).
    Return dict {group_id: [filepath, ...]}
    """
    import imagehash

    print("[find_duplicates] Menghitung pHash semua gambar...")
    hashes = []
    for fp in df["filepath"]:
        h = compute_phash(fp, hash_size)
        hashes.append(h)
    df = df.copy()
    df["phash"] = hashes

    # Group by exact hash dulu
    groups: Dict[str, List[str]] = {}
    if threshold == 0:
        for _, row in df.dropna(subset=["phash"]).iterrows():
            h = row["phash"]
            groups.setdefault(h, []).append(row["filepath"])
        groups = {k: v for k, v in groups.items() if len(v) > 1}
    else:
        # Near-duplicate via hamming distance (lebih lambat)
        valid = df.dropna(subset=["phash"]).copy()
        hash_objs = [imagehash.hex_to_hash(h) for h in valid["phash"]]
        visited = set()
        for i, (fp_i, h_i) in enumerate(zip(valid["filepath"], hash_objs)):
            if i in visited:
                continue
            group = [fp_i]
            for j, (fp_j, h_j) in enumerate(zip(valid["filepath"], hash_objs)):
                if i != j and j not in visited and (h_i - h_j) <= threshold:
                    group.append(fp_j)
                    visited.add(j)
            if len(group) > 1:
                groups[fp_i] = group
            visited.add(i)

    print(f"[find_duplicates] Grup duplikat ditemukan: {len(groups)}")
    return groups


def assign_group_ids(df: pd.DataFrame, dup_groups: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Tambahkan kolom 'group_id' ke df. File tanpa duplikat punya group_id unik.
    """
    fp_to_group = {}
    for gid, (key, members) in enumerate(dup_groups.items()):
        for fp in members:
            fp_to_group[fp] = gid

    df = df.copy()
    max_gid = len(dup_groups)
    df["group_id"] = df["filepath"].apply(
        lambda fp: fp_to_group.get(fp, max_gid + df[df["filepath"] == fp].index[0])
    )
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

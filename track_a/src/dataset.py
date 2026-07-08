"""
dataset.py — PyTorch Dataset & DataLoader
Track A — BDC Satria Data 2026

ARTEFAK UTAMA: File ini diserahkan ke Track B & C.
Jangan modifikasi setelah handoff (10 Juli pagi).
"""

from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ─── Label Mapping ────────────────────────────────────────────────────────────

LABEL_MAP = {
    "0_Recyclable": 0,
    "1_Electronic": 1,
    "2_Organic": 2,
}
CLASS_NAMES = ["Recyclable", "Electronic", "Organic"]

# Normalisasi ImageNet (wajib untuk pretrained backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = 224) -> T.Compose:
    """
    Augmentasi untuk training. Bisa dituning oleh Track B.
    """
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(img_size: int = 224) -> T.Compose:
    """
    Transform untuk validasi & test — tanpa augmentasi.
    Pakai Resize + CenterCrop (bukan Resize langsung ke square) untuk
    menghindari distorsi gambar dengan aspect ratio ekstrem (EDA: 0.5–6.0).
    """
    return T.Compose([
        T.Resize(img_size),           # resize sisi terpendek ke img_size
        T.CenterCrop(img_size),       # crop tengah → square tanpa distorsi
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Dataset Class ────────────────────────────────────────────────────────────

class WasteDataset(Dataset):
    """
    Dataset klasifikasi sampah BDC 2026.

    Args:
        df          : DataFrame dengan kolom 'filepath' dan (opsional) 'label'
        transform   : torchvision transform
        is_test     : True → tidak ada label, return (image, filepath)
        skip_list   : set filepath gambar corrupt yang akan di-skip (sudah dihandle via df)

    Returns (train/val mode):
        image (Tensor[C,H,W]), label (int)

    Returns (test mode):
        image (Tensor[C,H,W]), filepath (str)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        is_test: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

        if not is_test:
            # Pastikan label ada dan valid
            assert "label" in df.columns, "Kolom 'label' tidak ada di DataFrame"
            assert set(df["label"].unique()).issubset({0, 1, 2}), \
                f"Label harus 0/1/2, dapat: {df['label'].unique()}"
            print(f"[WasteDataset] Loaded {len(df)} sampel | "
                  f"dist: {dict(df['label'].value_counts().sort_index())}")
        else:
            print(f"[WasteDataset][TEST] Loaded {len(df)} gambar test")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fp = row["filepath"]

        try:
            img = Image.open(fp).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Gagal membaca gambar: {fp} — {e}")

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, fp
        else:
            label = int(row["label"])
            return img, label


# ─── DataLoader Factories ─────────────────────────────────────────────────────

def make_fold_loaders(
    folds_csv: str,
    val_fold: int,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    skip_filepaths: Optional[set] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Buat train & val DataLoader untuk satu fold dari folds.csv.

    Args:
        folds_csv    : path ke folds.csv (kolom: filepath, label, fold)
        val_fold     : nomor fold yang jadi validation (0–4)
        img_size     : resolusi input model
        batch_size   : ukuran batch
        num_workers  : worker DataLoader
        skip_filepaths: set filepath yang di-skip (dari skip-list EDA)

    Returns:
        train_loader, val_loader
    """
    df = pd.read_csv(folds_csv)

    # Validasi kolom
    assert all(c in df.columns for c in ["filepath", "label", "fold"]), \
        "folds.csv harus punya kolom: filepath, label, fold"

    # Filter skip-list
    if skip_filepaths:
        before = len(df)
        df = df[~df["filepath"].isin(skip_filepaths)].reset_index(drop=True)
        print(f"[make_fold_loaders] Skip {before - len(df)} gambar corrupt")

    df_train = df[df["fold"] != val_fold].reset_index(drop=True)
    df_val   = df[df["fold"] == val_fold].reset_index(drop=True)

    # Verifikasi no-overlap
    train_fps = set(df_train["filepath"])
    val_fps   = set(df_val["filepath"])
    overlap = train_fps & val_fps
    assert len(overlap) == 0, f"LEAKAGE TERDETEKSI! {len(overlap)} file ada di train & val"

    train_ds = WasteDataset(df_train, transform=get_train_transform(img_size))
    val_ds   = WasteDataset(df_val,   transform=get_eval_transform(img_size))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[make_fold_loaders] Fold {val_fold} → "
          f"train: {len(df_train)} | val: {len(df_val)}")
    return train_loader, val_loader

def get_loaders(fold, img_size=224, batch=32, folds_csv=None, num_workers=2):
    """
    Wrapper sesuai kontrak Workflow_Koordinasi_ABC.md.
    Signature: get_loaders(fold, img_size, batch) -> (train_loader, val_loader)
    
    Args:
        fold      : nomor fold validasi (0–4)
        img_size  : resolusi input (default 224)
        batch     : ukuran batch (default 32)
        folds_csv : path ke folds.csv — wajib diisi, atau set env FOLDS_CSV
        num_workers: worker DataLoader (default 2)
    """
    import os
    if folds_csv is None:
        folds_csv = os.environ.get("FOLDS_CSV")
    if not folds_csv:
        raise ValueError(
            "folds_csv harus diisi. Contoh:\n"
            "  get_loaders(0, folds_csv='/content/drive/MyDrive/BDC2026/folds.csv')\n"
            "  atau set env: os.environ['FOLDS_CSV'] = '...'"
        )
    return make_fold_loaders(
        folds_csv=folds_csv,
        val_fold=fold,
        img_size=img_size,
        batch_size=batch,
        num_workers=num_workers,
    )

def make_test_loader(
    test_dir: str,
    submission_csv: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
) -> DataLoader:
    """
    Buat test DataLoader dengan urutan PERSIS sesuai submission.csv.
    PENTING: Urutan ini wajib dipertahankan untuk submission yang benar.

    Args:
        test_dir       : direktori gambar test
        submission_csv : template submission.csv (menentukan urutan)
        img_size       : resolusi input
        batch_size     : ukuran batch
        num_workers    : worker DataLoader

    Returns:
        test_loader (setiap item: (image_tensor, filepath_str))
    """
    test_dir = Path(test_dir)
    sub_df = pd.read_csv(submission_csv)

    # Ambil nama file dari kolom pertama submission
    id_col = sub_df.columns[0]
    ordered_filenames = sub_df[id_col].astype(str).tolist()

    filepaths = []
    missing = []
    for fname in ordered_filenames:
        fp = test_dir / fname
        if fp.exists():
            filepaths.append(str(fp))
        else:
            # Coba dengan ekstensi umum
            found = False
            for ext in [".jpg", ".jpeg", ".png"]:
                fp_try = test_dir / (fname + ext)
                if fp_try.exists():
                    filepaths.append(str(fp_try))
                    found = True
                    break
            if not found:
                missing.append(fname)

    if missing:
        print(f"[make_test_loader] ⚠️  {len(missing)} file tidak ditemukan di test_dir")

    df_test = pd.DataFrame({"filepath": filepaths})
    test_ds = WasteDataset(df_test, transform=get_eval_transform(img_size), is_test=True)

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,  # ← JANGAN shuffle test!
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[make_test_loader] {len(df_test)} gambar test, urutan sesuai submission.csv")
    return test_loader


# ─── Quick Sanity Check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    folds_csv = sys.argv[1] if len(sys.argv) > 1 else "outputs/folds.csv"

    print("=" * 50)
    print("SANITY CHECK: WasteDataset & DataLoader")
    print("=" * 50)

    train_loader, val_loader = make_fold_loaders(
        folds_csv=folds_csv,
        val_fold=0,
        img_size=224,
        batch_size=8,
        num_workers=0,
    )

    # Cek 1 batch train
    imgs, labels = next(iter(train_loader))
    print(f"\n✅ Train batch — images: {imgs.shape}, labels: {labels.tolist()}")
    assert imgs.shape[1:] == (3, 224, 224), "Shape gambar salah!"
    assert all(l in {0, 1, 2} for l in labels.tolist()), "Label di luar 0/1/2!"

    # Cek 1 batch val
    imgs_v, labels_v = next(iter(val_loader))
    print(f"✅ Val batch   — images: {imgs_v.shape}, labels: {labels_v.tolist()}")

    print("\n✅ Semua assertion lolos! Dataset siap untuk Track B.")

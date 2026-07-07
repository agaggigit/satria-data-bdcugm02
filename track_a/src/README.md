# Track A — Panduan Modul `src/`

**BDC Satria Data 2026 — Klasifikasi Citra Sampah**

Folder ini berisi dua modul Python yang menjadi **artefak utama handoff** dari Track A ke Track B & C.
Kedua file ini adalah kode yang *sebenarnya dipakai saat training dan inference* — bukan hanya notebook eksplorasi.

> ⚠️ **Jangan modifikasi file ini setelah handoff (10 Juli pagi).**
> Kalau Track B perlu menyesuaikan augmentasi, diskusikan dulu dengan Track A
> agar seed dan konfigurasi tetap terdokumentasi.

---

## Daftar File

| File | Peran | Dipakai oleh |
|------|-------|--------------|
| [`dataset.py`](#datasetpy) | Dataset class, transform, factory DataLoader | Track B (training), Track C (inference) |
| [`utils.py`](#utilspy) | Helper: scan dir, cek corrupt, deteksi duplikat, class weights | Track A internal (notebook EDA & split) |

---

## `dataset.py`

File inti yang mengemas seluruh logika data loading ke dalam antarmuka PyTorch standar.
Track B cukup `import` file ini — tidak perlu menyentuh path gambar secara manual.

### Konstanta Global

```python
LABEL_MAP = {
    "0_Recyclable": 0,
    "1_Electronic": 1,
    "2_Organic":    2,
}
CLASS_NAMES   = ["Recyclable", "Electronic", "Organic"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # normalisasi wajib untuk pretrained backbone
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

---

### `get_train_transform(img_size=224)` → `T.Compose`

Transform yang diaplikasikan ke data **training**. Mengandung augmentasi untuk meningkatkan
generalisasi model.

| Augmentasi | Parameter | Catatan |
|-----------|-----------|---------|
| `RandomResizedCrop` | scale=(0.7, 1.0) | Crop acak lalu resize ke `img_size × img_size` |
| `RandomHorizontalFlip` | p=0.5 | Flip horizontal |
| `RandomVerticalFlip` | p=0.2 | Flip vertikal (jarang, tapi relevan untuk sampah dari atas) |
| `ColorJitter` | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05 | Variasi pencahayaan & warna |
| `RandomRotation` | 15° | Rotasi acak ±15 derajat |
| `ToTensor` + `Normalize` | ImageNet mean/std | Wajib untuk pretrained backbone |

**Contoh penggunaan:**
```python
from src.dataset import get_train_transform
transform = get_train_transform(img_size=224)
```

---

### `get_eval_transform(img_size=224)` → `T.Compose`

Transform untuk **validasi dan test** — tanpa augmentasi, hanya resize deterministik + normalisasi.
Pastikan selalu menggunakan fungsi ini (bukan train transform) saat evaluasi.

```python
from src.dataset import get_eval_transform
transform = get_eval_transform(img_size=224)
```

---

### `class WasteDataset(Dataset)`

Kelas dataset utama. Mewarisi `torch.utils.data.Dataset` dan kompatibel penuh
dengan `DataLoader` PyTorch.

**Argumen konstruktor:**

| Parameter | Tipe | Wajib | Deskripsi |
|-----------|------|-------|-----------|
| `df` | `pd.DataFrame` | ✅ | DataFrame dengan kolom `filepath` (+ `label` jika bukan test) |
| `transform` | `Callable` | ❌ | Transform torchvision; `None` = gambar dikembalikan sebagai PIL |
| `is_test` | `bool` | ❌ | `True` untuk set test (tidak ada label) |

**Return value `__getitem__`:**

| Mode | Return |
|------|--------|
| Train / Val (`is_test=False`) | `(image: Tensor[3, H, W], label: int)` |
| Test (`is_test=True`) | `(image: Tensor[3, H, W], filepath: str)` |

**Validasi otomatis saat init:**
- Mode train/val: assert kolom `label` ada dan nilainya subset dari `{0, 1, 2}`
- Saat `__getitem__`: buka gambar dengan PIL, konversi ke RGB; raise `RuntimeError` jika gagal

**Contoh penggunaan:**
```python
import pandas as pd
from src.dataset import WasteDataset, get_train_transform

df = pd.read_csv("outputs/folds.csv")
df_fold0_val = df[df["fold"] == 0]

dataset = WasteDataset(df_fold0_val, transform=get_train_transform(224))
img, label = dataset[0]   # Tensor[3, 224, 224], int
print(img.shape, label)   # torch.Size([3, 224, 224])  1
```

---

### `make_fold_loaders(folds_csv, val_fold, ...)` → `(DataLoader, DataLoader)`

Factory function untuk membuat pasangan train + val `DataLoader` dari satu fold.
Fungsi ini yang **dipanggil di dalam training loop Track B**.

**Argumen:**

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `folds_csv` | — | Path ke `folds.csv` (wajib) |
| `val_fold` | — | Nomor fold validation: `0`, `1`, `2`, `3`, atau `4` (wajib) |
| `img_size` | `224` | Resolusi input; harus konsisten dengan arsitektur model |
| `batch_size` | `32` | Ukuran batch |
| `num_workers` | `2` | Worker parallel DataLoader; set `0` di Windows lokal jika ada error |
| `skip_filepaths` | `None` | `set` filepath corrupt dari skip-list EDA; otomatis difilter |

**Jaminan keamanan yang built-in:**
- Validasi kolom `folds.csv` ada (`filepath`, `label`, `fold`)
- Assert no-overlap antara train dan val set — akan `AssertionError` jika ada leakage
- Train loader: `shuffle=True`, `drop_last=True`
- Val loader: `shuffle=False`

**Contoh penggunaan di Track B:**
```python
from src.dataset import make_fold_loaders

train_loader, val_loader = make_fold_loaders(
    folds_csv="track_a/outputs/folds.csv",
    val_fold=0,
    img_size=224,
    batch_size=32,
    num_workers=4,
)

for imgs, labels in train_loader:
    # imgs: Tensor[batch, 3, 224, 224]
    # labels: Tensor[batch]  (nilai 0, 1, atau 2)
    ...
```

---

### `make_test_loader(test_dir, submission_csv, ...)` → `DataLoader`

Factory function untuk membuat test `DataLoader` dengan urutan gambar yang **dijamin
sesuai `submission.csv`**. Ini krusial — urutan yang salah menghasilkan submission yang salah.

**Argumen:**

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `test_dir` | — | Direktori gambar test (wajib) |
| `submission_csv` | — | Template `submission.csv` sebagai penentu urutan (wajib) |
| `img_size` | `224` | Resolusi input |
| `batch_size` | `32` | Ukuran batch |
| `num_workers` | `2` | Worker DataLoader |

**Perilaku penting:**
- Urutan gambar diambil dari kolom pertama `submission.csv` — **`shuffle=False` selalu**
- Otomatis mencoba fallback ekstensi (`.jpg`, `.jpeg`, `.png`) jika nama file tanpa ekstensi
- Setiap item return: `(image_tensor, filepath_str)` — filepath dipakai Track C untuk menyusun submission

**Contoh penggunaan di Track C:**
```python
from src.dataset import make_test_loader

test_loader = make_test_loader(
    test_dir="datasets/BDC2026/test",
    submission_csv="datasets/BDC2026/submission.csv",
    img_size=224,
    batch_size=64,
    num_workers=4,
)

for imgs, filepaths in test_loader:
    preds = model(imgs.to(device)).argmax(dim=1)
    # preds terurut sesuai submission.csv
```

---

### Sanity Check (`__main__`)

`dataset.py` bisa dijalankan langsung sebagai script untuk verifikasi cepat:

```bash
# Dari folder track_a/
python src/dataset.py outputs/folds.csv
```

Ini akan:
1. Membuat train & val loader untuk fold 0 (batch_size=8)
2. Mengambil 1 batch dari keduanya
3. Assert shape `[batch, 3, 224, 224]` dan label valid `{0, 1, 2}`
4. Print hasil — jika tidak ada error, DataLoader siap dipakai

---

## `utils.py`

Kumpulan fungsi helper yang dipakai oleh notebook EDA dan split.
Umumnya **tidak diimpor langsung oleh Track B/C** — fungsi-fungsi ini sudah
terintegrasi di dalam notebook Track A.

### Konstanta Global

```python
LABEL_MAP    = {"0_Recyclable": 0, "1_Electronic": 1, "2_Organic": 2}
IDX_TO_CLASS = {0: "0_Recyclable", 1: "1_Electronic", 2: "2_Organic"}  # inverse
CLASS_NAMES  = ["Recyclable", "Electronic", "Organic"]
```

---

### Dataset Scanning

#### `scan_train_dir(train_dir)` → `pd.DataFrame`

Scan folder `train/` dan kumpulkan semua path gambar beserta labelnya.
Menjamin mapping folder → label integer sesuai `LABEL_MAP`.

```python
from src.utils import scan_train_dir
df = scan_train_dir("datasets/BDC2026/train")
# Kolom: filepath (str), label_name (str), label (int)
```

Output DataFrame:

| filepath | label_name | label |
|----------|-----------|-------|
| `.../0_Recyclable/img001.jpg` | `0_Recyclable` | `0` |
| `.../1_Electronic/img002.jpg` | `1_Electronic` | `1` |

#### `scan_test_dir(test_dir)` → `pd.DataFrame`

Scan folder `test/` dan urutkan file secara alfanumerik.
Kolom output: `filepath`, `filename`.

> **Catatan:** Untuk test loader yang benar-benar terurut sesuai `submission.csv`,
> gunakan `make_test_loader()` di `dataset.py`, bukan fungsi ini.

---

### Image Health Checks

#### `check_corrupt(df, verbose=True)` → `(pd.DataFrame, List[str])`

Iterasi seluruh filepath di `df`, coba buka + verify tiap gambar dengan PIL.
Gambar yang gagal dimasukkan ke `skip_list`.

```python
from src.utils import check_corrupt
df_clean, skip_list = check_corrupt(df)
# df_clean: DataFrame tanpa gambar corrupt
# skip_list: list filepath yang bermasalah
```

> ⏱️ Memakan waktu O(n) — untuk 26k gambar butuh ~5–10 menit.

#### `get_image_stats(df, sample_n=None)` → `pd.DataFrame`

Kumpulkan width, height, channels, dan aspect_ratio tiap gambar.
Set `sample_n` untuk sampling acak agar lebih cepat.

```python
from src.utils import get_image_stats
stats_df = get_image_stats(df, sample_n=2000)
# Kolom: filepath, width, height, channels, aspect_ratio
print(stats_df["width"].describe())
```

---

### Duplicate Detection

#### `compute_phash(filepath, hash_size=8)` → `str | None`

Hitung perceptual hash (pHash) satu gambar. Return `None` jika gambar gagal dibuka.

```python
from src.utils import compute_phash
h = compute_phash("path/to/image.jpg")   # contoh: "f8e0c0a0b0d0e0f0"
```

#### `find_duplicates(df, hash_size=8, threshold=5)` → `Dict[str, List[str]]`

Deteksi duplikat/near-duplikat di seluruh DataFrame menggunakan pHash.

| Parameter | Deskripsi |
|-----------|-----------|
| `hash_size` | Presisi hash — lebih besar lebih sensitif; default `8` sudah cukup |
| `threshold` | Hamming distance maksimum yang dianggap duplikat; `0` = identik saja, `>0` = near-duplicate |

```python
from src.utils import find_duplicates
groups = find_duplicates(df, threshold=0)   # hanya exact duplicate
# groups = {"filepath_A": ["filepath_A", "filepath_B"], ...}
```

> ⏱️ `threshold=0` (exact): cepat, O(n). `threshold>0` (near-dup): lambat, O(n²).
> Untuk dataset 26k gambar, gunakan `threshold=0` dulu.

#### `assign_group_ids(df, dup_groups)` → `pd.DataFrame`

Tambahkan kolom `group_id` ke DataFrame berdasarkan hasil `find_duplicates()`.
Gambar dalam grup duplikat mendapat `group_id` yang sama —
dipakai oleh `StratifiedGroupKFold` agar kembaran tidak terpisah antar fold.

```python
from src.utils import assign_group_ids
df = assign_group_ids(df, groups)
# Kolom baru: group_id (int)
```

---

### Class Weights

#### `compute_class_weights(labels, n_classes=3)` → `np.ndarray`

Hitung class weights dengan formula:
```
weight[c] = n_samples / (n_classes × count[c])
```

Menghasilkan vektor bobot `[w0, w1, w2]` yang bisa langsung dipakai di
`torch.nn.CrossEntropyLoss(weight=...)`.

```python
from src.utils import compute_class_weights
weights = compute_class_weights(df["label"].tolist())
# array([0.89, 1.45, 0.76])  ← contoh, nilai sesuai distribusi dataset

# Cara pakai di Track B:
import torch, torch.nn as nn
w_tensor = torch.tensor(weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=w_tensor)
```

---

### I/O Helpers

#### `save_eda_stats(stats, path="outputs/eda_stats.json")`

Simpan dictionary statistik EDA ke file JSON dengan format rapi (indent=2).
Membuat parent directory secara otomatis.

#### `save_class_weights(weights, path="outputs/class_weights.npy")`

Simpan array numpy ke file `.npy`. Track B load dengan:
```python
weights = np.load("track_a/outputs/class_weights.npy")
```

---

## Cara Import dari Notebook / Script Lain

Jika menjalankan dari luar folder `track_a/`, tambahkan path ke `sys.path`:

```python
import sys
sys.path.append("track_a/src")           # dari root repo
# atau jika di Colab:
sys.path.append("/content/track_a/src")  # sesuaikan path

from dataset import make_fold_loaders, make_test_loader, WasteDataset
from utils import compute_class_weights, scan_train_dir
```

Atau jika menjalankan langsung dari folder `track_a/`:

```python
from src.dataset import make_fold_loaders
from src.utils import compute_class_weights
```

---

## Ringkasan: Siapa Pakai Apa

| Fungsi / Class | Track A | Track B | Track C |
|----------------|:-------:|:-------:|:-------:|
| `scan_train_dir` | ✅ (notebook) | — | — |
| `check_corrupt` | ✅ (notebook) | — | — |
| `get_image_stats` | ✅ (notebook) | — | — |
| `find_duplicates` | ✅ (notebook) | — | — |
| `compute_class_weights` | ✅ (notebook) | ✅ (load `.npy`) | — |
| `get_train_transform` | ✅ (via loader) | ✅ (via loader) | — |
| `get_eval_transform` | ✅ (via loader) | ✅ (via loader) | ✅ (via loader) |
| `WasteDataset` | ✅ (via loader) | ✅ (via loader) | ✅ (via loader) |
| `make_fold_loaders` | ✅ (verifikasi) | ✅ **utama** | — |
| `make_test_loader` | ✅ (verifikasi) | — | ✅ **utama** |

---

*Dibuat untuk BDC Satria Data 2026 — Track A | Last updated: 8 Juli 2026*
